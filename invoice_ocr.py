# invoice_ocr_zoho.py
# Targeted extraction for Zoho-style Tax Invoice layout

import cv2
import pytesseract
import re
import json
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ─────────────────────────────────────────
# STEP 1: Preprocessing
# ─────────────────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2
    )
    return thresh


# ─────────────────────────────────────────
# STEP 2: OCR
# ─────────────────────────────────────────
def extract_text(processed_img: np.ndarray) -> str:
    config = r'--oem 1 --psm 6'
    return pytesseract.image_to_string(processed_img, config=config)


# ─────────────────────────────────────────
# HELPER: extract address block after a label
# ─────────────────────────────────────────
def extract_address_block(text: str, start_marker: str, end_markers: list) -> dict:
    """
    Extracts a multi-line address block starting after start_marker
    and stopping before any of the end_markers.
    Returns parsed address components.
    """
    pattern = re.escape(start_marker)
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return {}

    block = text[m.end():]

    # Cut off at end markers
    for em in end_markers:
        cut = re.search(re.escape(em), block, re.IGNORECASE)
        if cut:
            block = block[:cut.start()]

    lines = [l.strip() for l in block.strip().splitlines() if l.strip()]

    result = {}
    if not lines:
        return result

    result["name"] = lines[0] if len(lines) > 0 else None
    result["street"] = lines[1] if len(lines) > 1 else None
    result["city"] = lines[2] if len(lines) > 2 else None
    result["state"] = lines[3] if len(lines) > 3 else None
    result["country"] = lines[4] if len(lines) > 4 else None

    # GSTIN: look for "GSTIN XXXXXXXXX" pattern inside block
    gstin = re.search(r'GSTIN\s+([A-Z0-9]+)', block, re.IGNORECASE)
    if gstin:
        result["gstin"] = gstin.group(1).strip()
        # Remove GSTIN line from country if it got mixed in
        if result.get("country") and "gstin" in result["country"].lower():
            result["country"] = lines[4] if len(lines) > 5 else None

    return result


# ─────────────────────────────────────────
# STEP 3: Targeted Field Extraction
# ─────────────────────────────────────────
def extract_fields(text: str) -> dict:

    result = {
        "invoice_type":       None,
        "invoice_number":     None,
        "invoice_date":       None,
        "due_date":           None,
        "place_of_supply":    None,

        "vendor": {
            "name":    None,
            "contact": None,
            "address": None,
            "city":    None,
            "state":   None,
            "country": None,
            "gstin":   None,
        },

        "bill_to": {
            "name":    None,
            "street":  None,
            "city":    None,
            "state":   None,
            "country": None,
            "gstin":   None,
        },

        "line_items": [],

        "subtotal":    None,
        "igst_rate":   None,
        "igst_amount": None,
        "cess_rate":   None,
        "cess_amount": None,
        "total":       None,
        "currency":    None,

        "notes":             None,
        "terms_conditions":  None,
    }

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # ── Invoice Type ─────────────────────────────────────────────────
    m = re.search(r'(TAX INVOICE|INVOICE|PROFORMA INVOICE|CREDIT NOTE|DEBIT NOTE)', text, re.IGNORECASE)
    if m:
        result["invoice_type"] = m.group(1).strip().upper()

    # ── Invoice Number ───────────────────────────────────────────────
    m = re.search(r'Invoice\s*#\s*([A-Z0-9\-/]+)', text, re.IGNORECASE)
    if m:
        result["invoice_number"] = m.group(1).strip()

    # ── Invoice Date ─────────────────────────────────────────────────
    m = re.search(
        r'Invoice\s*Date\s*[:\s]*'
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}'
        r'|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        text, re.IGNORECASE
    )
    if m:
        result["invoice_date"] = m.group(1).strip()

    # ── Due Date ─────────────────────────────────────────────────────
    m = re.search(
        r'Due\s*Date\s*[:\s]*'
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}'
        r'|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        text, re.IGNORECASE
    )
    if m:
        result["due_date"] = m.group(1).strip()

    # ── Place of Supply ──────────────────────────────────────────────
    m = re.search(r'Place\s*of\s*Supply\s*[:\s]*([A-Za-z\s]+)', text, re.IGNORECASE)
    if m:
        result["place_of_supply"] = m.group(1).strip()

    # ── Vendor Block (top-left, before "Bill To") ─────────────────────
    # Everything before "Bill To" is vendor info
    bill_to_pos = re.search(r'Bill\s*To', text, re.IGNORECASE)
    vendor_block = text[:bill_to_pos.start()] if bill_to_pos else ""

    vendor_lines = [l.strip() for l in vendor_block.splitlines() if l.strip()]
    # Remove "TAX INVOICE" and "Invoice#" lines from vendor block
    vendor_lines = [
        l for l in vendor_lines
        if not re.search(r'tax invoice|invoice\s*#|INV-', l, re.IGNORECASE)
    ]

    if vendor_lines:
        result["vendor"]["name"]    = vendor_lines[0] if len(vendor_lines) > 0 else None
        result["vendor"]["contact"] = vendor_lines[1] if len(vendor_lines) > 1 else None
        result["vendor"]["address"] = vendor_lines[2] if len(vendor_lines) > 2 else None
        result["vendor"]["city"]    = vendor_lines[3] if len(vendor_lines) > 3 else None
        result["vendor"]["state"]   = vendor_lines[4] if len(vendor_lines) > 4 else None
        result["vendor"]["country"] = vendor_lines[5] if len(vendor_lines) > 5 else None
        gstin_v = re.search(r'GSTIN\s+([A-Z0-9]+)', vendor_block, re.IGNORECASE)
        if gstin_v:
            result["vendor"]["gstin"] = gstin_v.group(1).strip()

    # ── Bill To Block ────────────────────────────────────────────────
    # From "Bill To" until "Invoice Date" or "Place of Supply"
    if bill_to_pos:
        end_markers = ["Invoice Date", "Due Date", "Place of Supply"]
        bt = extract_address_block(text, "Bill To", end_markers)
        result["bill_to"]["name"]    = bt.get("name")
        result["bill_to"]["street"]  = bt.get("street")
        result["bill_to"]["city"]    = bt.get("city")
        result["bill_to"]["state"]   = bt.get("state")
        result["bill_to"]["country"] = bt.get("country")
        result["bill_to"]["gstin"]   = bt.get("gstin")

    # ── Currency ─────────────────────────────────────────────────────
    m = re.search(r'Rs\.?|₹|\bINR\b|\bUSD\b|\bEUR\b', text)
    if m:
        sym = m.group(0).upper()
        result["currency"] = "INR" if sym in ("RS.", "RS", "₹") else sym

    # ── Line Items ───────────────────────────────────────────────────
    # Table columns: # | Item Description | HSN/SAC | Qty | Rate | IGST | Cess | Amount
    # Find the table body: rows that start with a row number digit
    item_pattern = re.compile(
        r'^\s*(\d+)\s+'                         # row number
        r'([A-Za-z][\w\s,.\-()]{2,50?})\s+'    # description
        r'(\d*)\s+'                              # HSN/SAC (may be blank)
        r'(\d+(?:\.\d+)?)\s+'                   # qty
        r'([\d,]+\.?\d*)\s+'                    # rate
        r'([\d,]+\.?\d*)\s+'                    # igst amount
        r'(\d+)\s+'                              # igst %
        r'([\d,]+\.?\d*)\s+'                    # cess amount
        r'(\d+)\s+'                              # cess %
        r'([\d,]+\.?\d*)',                       # line total
        re.MULTILINE
    )
    for m in item_pattern.finditer(text):
        result["line_items"].append({
            "sr_no":       m.group(1).strip(),
            "description": m.group(2).strip(),
            "hsn_sac":     m.group(3).strip() or None,
            "quantity":    m.group(4).strip(),
            "rate":        m.group(5).replace(',', '').strip(),
            "igst_amount": m.group(6).replace(',', '').strip(),
            "igst_pct":    m.group(7).strip() + "%",
            "cess_amount": m.group(8).replace(',', '').strip(),
            "cess_pct":    m.group(9).strip() + "%",
            "amount":      m.group(10).replace(',', '').strip(),
        })

    # Fallback: simpler line item pattern if above finds nothing
    if not result["line_items"]:
        simple = re.compile(
            r'^\s*\d+\s+([A-Za-z][\w\s]{2,40})\s+(\d+(?:\.\d+)?)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)',
            re.MULTILINE
        )
        for m in simple.finditer(text):
            desc, qty, rate, amount = m.groups()
            if desc.strip().lower() not in {"sub total", "total", "igst", "cess"}:
                result["line_items"].append({
                    "description": desc.strip(),
                    "quantity": qty.strip(),
                    "rate": rate.replace(',', '').strip(),
                    "amount": amount.replace(',', '').strip(),
                })

    # ── Subtotal ─────────────────────────────────────────────────────
    m = re.search(r'Sub\s*Total\s+([\d,]+\.?\d*)', text, re.IGNORECASE)
    if m:
        result["subtotal"] = m.group(1).replace(',', '').strip()

    # ── IGST ─────────────────────────────────────────────────────────
    m = re.search(r'IGST\s*\((\d+)%\)\s+([\d,]+\.?\d*)', text, re.IGNORECASE)
    if m:
        result["igst_rate"]   = m.group(1).strip() + "%"
        result["igst_amount"] = m.group(2).replace(',', '').strip()

    # ── Cess ─────────────────────────────────────────────────────────
    m = re.search(r'Cess\s*\((\d+)%\)\s+([\d,]+\.?\d*)', text, re.IGNORECASE)
    if m:
        result["cess_rate"]   = m.group(1).strip() + "%"
        result["cess_amount"] = m.group(2).replace(',', '').strip()

    # ── Total ────────────────────────────────────────────────────────
    m = re.search(r'TOTAL\s+(?:Rs\.?|₹)?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if m:
        result["total"] = m.group(1).replace(',', '').strip()

    # ── Notes ────────────────────────────────────────────────────────
    m = re.search(r'Notes?\s*\n([^\n]+(?:\n(?!Terms)[^\n]+)*)', text, re.IGNORECASE)
    if m:
        result["notes"] = m.group(1).strip()

    # ── Terms & Conditions ───────────────────────────────────────────
    m = re.search(r'Terms\s*[&and]+\s*Conditions?\s*\n([^\n]+(?:\n[^\n]+){0,3})', text, re.IGNORECASE)
    if m:
        result["terms_conditions"] = m.group(1).strip()

    return result


# ─────────────────────────────────────────
# STEP 4: Pipeline
# ─────────────────────────────────────────
def run_pipeline(image_path: str, debug: bool = False) -> dict:
    print(f"[1/3] Preprocessing: {image_path}")
    processed = preprocess_image(image_path)

    if debug:
        cv2.imwrite("debug_preprocessed.png", processed)
        print("      Saved debug_preprocessed.png")

    print("[2/3] Running Tesseract OCR...")
    raw_text = extract_text(processed)

    if debug:
        print("\n--- RAW OCR TEXT ---")
        print(raw_text)
        print("--------------------\n")

    print("[3/3] Extracting structured fields...")
    fields = extract_fields(raw_text)

    return fields


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_invoice.png"
    debug_mode = "--debug" in sys.argv

    result = run_pipeline(image_path, debug=debug_mode)

    print("\n✅ Extracted Invoice Data:")
    print(json.dumps(result, indent=2))

    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to output.json")