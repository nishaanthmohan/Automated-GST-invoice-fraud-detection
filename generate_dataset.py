"""
GST Invoice Dataset Generator with Ground-Truth Fraud Typologies
Compliant with Indian CBIC GST standards and tax slabs.
"""

import os
import random
import string
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Indian State Codes for GST
STATE_CODES = [
    "01", "02", "03", "04", "06", "07", "08", "09", "10",
    "18", "19", "23", "24", "27", "29", "32", "33", "36", "37"
]

VENDOR_PROFILES = [
    ("Tata Enterprises Ltd", "27AAACT1234F1Z5", "27", "Manufacturing", 450000),
    ("Reliance Logistics Solutions", "24AABCR5678H1Z2", "24", "Transport", 250000),
    ("Infosys IT Services", "29AAACI3456K1Z9", "29", "Software/Services", 850000),
    ("Apex Agro Traders", "33AABCA9876P1Z3", "33", "Agriculture", 120000),
    ("Shree Steel Traders", "27AABCS4321Q1Z4", "27", "Steel & Metals", 650000),
    ("Zenith Electronics", "07AABCZ8765L1Z1", "07", "Electronics", 320000),
    ("Global Pharma Chem", "24AABCG1122M1Z8", "24", "Pharmaceuticals", 420000),
    ("Vardhman Textiles", "03AABCV3344J1Z7", "03", "Textiles", 180000),
    ("Supreme Heavy Engineering", "27AABCS9988D1Z6", "27", "Engineering", 950000),
    ("Green Energy Infra", "36AABCG5566E1Z0", "36", "Energy", 780000),
    ("Kaveri Food Products", "33AABCK7788R1Z1", "33", "Food Products", 95000),
    ("Bharat Fasteners & Hardware", "27AABCB9900B1Z8", "27", "Hardware", 140000),
]

BUYER_PROFILES = [
    ("Godrej Consumer Ltd", "27AAACG1234D1Z2", "27"),
    ("Adani Port Logistics", "24AAACA5678B1Z7", "24"),
    ("Wipro Technologies", "29AAACW3456F1Z1", "29"),
    ("Larsen & Toubro Construction", "27AAACL9876H1Z8", "27"),
    ("Apollo Hospitals Enterprise", "33AAACA4321J1Z5", "33"),
    ("Mahindra Auto Components", "27AAACM8765K1Z3", "27"),
    ("ITC Hospitality & Foods", "19AAACI1122L1Z0", "19"),
    ("Sun Pharma Industries", "24AAACS3344N1Z9", "24"),
]

GST_RATES = [0.0, 0.05, 0.12, 0.18, 0.28]
RATE_PROBABILITIES = [0.05, 0.20, 0.25, 0.40, 0.10]

ITEMS_BY_RATE = {
    0.00: ["Fresh Vegetables", "Grains & Cereals", "Fresh Milk", "Salt"],
    0.05: ["Edible Oil", "Sugar", "Spices", "Transport Services", "Renewable Energy Devices"],
    0.12: ["Computers & Hardware", "Processed Food", "Apparel > 1000", "Business Consulting"],
    0.18: ["Software Licensing", "Steel Fabrications", "Electronics", "Telecom Services", "Capital Goods"],
    0.28: ["Automobile Parts", "Cement", "Luxury Consumer Goods", "Air Conditioners"]
}

def generate_random_gstin(state_code=None):
    if not state_code:
        state_code = random.choice(STATE_CODES)
    pan_chars = "".join(random.choices(string.ascii_uppercase, k=5))
    pan_nums = "".join(random.choices(string.digits, k=4))
    pan_last = random.choice(string.ascii_uppercase)
    entity_num = random.choice("1234")
    return f"{state_code}{pan_chars}{pan_nums}{pan_last}{entity_num}Z{random.choice(string.ascii_uppercase + string.digits)}"

def generate_gst_dataset(num_records=2500, anomaly_ratio=0.08, output_path="data/gst_invoices.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records = []
    
    start_date = datetime(2024, 1, 1)
    num_anomalies = int(num_records * anomaly_ratio)
    num_normal = num_records - num_anomalies

    # 1. Generate Normal Invoices
    for i in range(num_normal):
        vendor_name, seller_gstin, seller_state, sector, base_mean = random.choice(VENDOR_PROFILES)
        buyer_name, buyer_gstin, buyer_state = random.choice(BUYER_PROFILES)
        
        # Realistic date and regular business hours (8 AM to 8 PM)
        days_offset = random.randint(0, 240)
        hour = random.randint(8, 20)
        minute = random.randint(0, 59)
        invoice_datetime = start_date + timedelta(days=days_offset, hours=hour, minutes=minute)
        
        # Invoice number pattern
        inv_num = f"INV/{seller_state}/{invoice_datetime.strftime('%y%m')}/{1000 + i}"
        
        # Taxable amount following log-normal distribution around vendor base mean
        taxable_amount = round(float(np.random.normal(base_mean, base_mean * 0.22)), 2)
        taxable_amount = max(1500.0, taxable_amount)
        
        # GST Rate and standard tax breakdown
        gst_rate = float(np.random.choice(GST_RATES, p=RATE_PROBABILITIES))
        item_desc = random.choice(ITEMS_BY_RATE[gst_rate])
        
        is_interstate = (seller_state != buyer_state)
        total_tax = round(taxable_amount * gst_rate, 2)
        
        if is_interstate:
            igst = total_tax
            cgst = 0.0
            sgst = 0.0
        else:
            igst = 0.0
            cgst = round(total_tax / 2.0, 2)
            sgst = round(total_tax - cgst, 2)
            
        total_amount = round(taxable_amount + cgst + sgst + igst, 2)
        
        records.append({
            "invoice_number": inv_num,
            "invoice_date": invoice_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "seller_gstin": seller_gstin,
            "seller_name": vendor_name,
            "seller_state": seller_state,
            "buyer_gstin": buyer_gstin,
            "buyer_name": buyer_name,
            "buyer_state": buyer_state,
            "item_description": item_desc,
            "taxable_amount": taxable_amount,
            "gst_rate": gst_rate,
            "cgst_amount": cgst,
            "sgst_amount": sgst,
            "igst_amount": igst,
            "total_tax": total_tax,
            "total_amount": total_amount,
            "payment_mode": random.choice(["NEFT", "RTGS", "Bank Transfer", "UPI"]),
            "is_anomaly": 0,
            "fraud_category": "Normal",
            "anomaly_explanation_ground_truth": "Standard compliant invoice within normal operating parameters."
        })

    # 2. Inject Fraudulent / Anomalous Invoices
    fraud_types = [
        "TAX_CALC_MISMATCH",
        "SHELL_BURST_SPIKE",
        "ROUND_AMOUNT_SURGE",
        "OFF_HOUR_DUMPING",
        "INVALID_GSTIN",
        "DUPLICATE_INVOICE",
        "CIRCULAR_TRADING"
    ]
    
    for j in range(num_anomalies):
        fraud_type = random.choice(fraud_types)
        vendor_name, seller_gstin, seller_state, sector, base_mean = random.choice(VENDOR_PROFILES)
        buyer_name, buyer_gstin, buyer_state = random.choice(BUYER_PROFILES)
        
        days_offset = random.randint(0, 240)
        hour = random.randint(9, 18)
        minute = random.randint(0, 59)
        invoice_datetime = start_date + timedelta(days=days_offset, hours=hour, minutes=minute)
        inv_num = f"INV/{seller_state}/{invoice_datetime.strftime('%y%m')}/{9000 + j}"
        
        taxable_amount = round(float(np.random.normal(base_mean, base_mean * 0.2)), 2)
        taxable_amount = max(5000.0, taxable_amount)
        gst_rate = float(np.random.choice(GST_RATES[1:], p=[0.25, 0.3, 0.35, 0.1]))
        item_desc = random.choice(ITEMS_BY_RATE[gst_rate])
        is_interstate = (seller_state != buyer_state)
        
        explanation = ""
        
        if fraud_type == "TAX_CALC_MISMATCH":
            # Inflate tax claim artificially by 30% to 80% to claim excess Input Tax Credit
            declared_tax = round(taxable_amount * (gst_rate + random.choice([0.10, 0.15, 0.20])), 2)
            if is_interstate:
                igst = declared_tax
                cgst, sgst = 0.0, 0.0
            else:
                igst = 0.0
                cgst = round(declared_tax / 2.0, 2)
                sgst = round(declared_tax - cgst, 2)
            total_tax = cgst + sgst + igst
            total_amount = round(taxable_amount + total_tax, 2)
            explanation = f"Calculated GST tax ({total_tax:,.2f}) significantly mismatches expected slab rate {int(gst_rate*100)}% ({taxable_amount * gst_rate:,.2f})."
            
        elif fraud_type == "SHELL_BURST_SPIKE":
            # Dormant / shell entity suddenly issues invoice 12x to 25x above normal turnover
            taxable_amount = round(base_mean * random.uniform(12.0, 28.0), 2)
            total_tax = round(taxable_amount * gst_rate, 2)
            if is_interstate:
                igst, cgst, sgst = total_tax, 0.0, 0.0
            else:
                igst, cgst = 0.0, round(total_tax / 2.0, 2)
                sgst = round(total_tax - cgst, 2)
            total_amount = round(taxable_amount + total_tax, 2)
            explanation = f"Abnormal volume surge: Invoice amount (₹{total_amount:,.2f}) is over 15x vendor historical mean."
            
        elif fraud_type == "ROUND_AMOUNT_SURGE":
            # Suspicious flat round amounts (e.g. 50,00,000, 1,00,00,000) typical of paper billing
            taxable_amount = float(random.choice([2500000.0, 5000000.0, 7500000.0, 10000000.0]))
            gst_rate = 0.18
            total_tax = round(taxable_amount * gst_rate, 2)
            igst = total_tax if is_interstate else 0.0
            cgst = 0.0 if is_interstate else round(total_tax / 2.0, 2)
            sgst = 0.0 if is_interstate else round(total_tax - cgst, 2)
            total_amount = round(taxable_amount + total_tax, 2)
            explanation = f"High-risk circular billing pattern: Exact round invoice value ₹{taxable_amount:,.0f} with standard 18% ITC pass-through."
            
        elif fraud_type == "OFF_HOUR_DUMPING":
            # Issued between 01:00 AM and 04:30 AM right before month end
            hour = random.choice([1, 2, 3, 4])
            invoice_datetime = invoice_datetime.replace(hour=hour, minute=random.randint(5, 55))
            total_tax = round(taxable_amount * gst_rate, 2)
            igst = total_tax if is_interstate else 0.0
            cgst = 0.0 if is_interstate else round(total_tax / 2.0, 2)
            sgst = 0.0 if is_interstate else round(total_tax - cgst, 2)
            total_amount = round(taxable_amount + total_tax, 2)
            explanation = f"Atypical timestamp: Invoice generated at {invoice_datetime.strftime('%H:%M:%S')} during off-hour batch dump."
            
        elif fraud_type == "INVALID_GSTIN":
            # Invalid state code or corrupted checksum
            corrupted_gstin = "99" + seller_gstin[2:14] + "!"
            seller_gstin = corrupted_gstin
            total_tax = round(taxable_amount * gst_rate, 2)
            igst = total_tax if is_interstate else 0.0
            cgst = 0.0 if is_interstate else round(total_tax / 2.0, 2)
            sgst = 0.0 if is_interstate else round(total_tax - cgst, 2)
            total_amount = round(taxable_amount + total_tax, 2)
            explanation = f"Syntax & Checksum failure: Seller GSTIN '{seller_gstin}' is non-compliant with CBIC 15-digit structure."
            
        elif fraud_type == "DUPLICATE_INVOICE":
            # Reuse existing invoice number with different or modified date
            if len(records) > 10:
                donor = random.choice(records[:100])
                inv_num = donor["invoice_number"]
                seller_gstin = donor["seller_gstin"]
                seller_name = donor["seller_name"]
            total_tax = round(taxable_amount * gst_rate, 2)
            igst = total_tax if is_interstate else 0.0
            cgst = 0.0 if is_interstate else round(total_tax / 2.0, 2)
            sgst = 0.0 if is_interstate else round(total_tax - cgst, 2)
            total_amount = round(taxable_amount + total_tax, 2)
            explanation = f"Duplicate invoice identifier: '{inv_num}' detected with multiple contradictory filings."
            
        elif fraud_type == "CIRCULAR_TRADING":
            # Reciprocal billing where buyer and seller exchange identical high values in cross loops
            taxable_amount = 4500000.0
            gst_rate = 0.18
            total_tax = round(taxable_amount * gst_rate, 2)
            igst = total_tax
            cgst, sgst = 0.0, 0.0
            total_amount = round(taxable_amount + total_tax, 2)
            explanation = "Circular Trading indicator: Rapid matched value loop across inter-state counterparty pair without tangible consumption."

        records.append({
            "invoice_number": inv_num,
            "invoice_date": invoice_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "seller_gstin": seller_gstin,
            "seller_name": vendor_name,
            "seller_state": seller_state,
            "buyer_gstin": buyer_gstin,
            "buyer_name": buyer_name,
            "buyer_state": buyer_state,
            "item_description": item_desc,
            "taxable_amount": taxable_amount,
            "gst_rate": gst_rate,
            "cgst_amount": cgst,
            "sgst_amount": sgst,
            "igst_amount": igst,
            "total_tax": total_tax,
            "total_amount": total_amount,
            "payment_mode": random.choice(["Cash", "NEFT", "RTGS"]),
            "is_anomaly": 1,
            "fraud_category": fraud_type,
            "anomaly_explanation_ground_truth": explanation
        })
        
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} GST invoices ({num_anomalies} anomalous) saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_gst_dataset()
