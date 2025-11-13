import pandas as pd
import numpy as np
from datetime import datetime
import os

def analyze_completeness(df):
    total = len(df)
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / total * 100).round(2)
    completeness = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_percent": missing_pct,
        "non_missing_count": total - missing_counts
    }).sort_values("missing_percent", ascending=False)
    return completeness

def analyze_consistency(df):
    issues = {}
    df2 = df.copy()
    today = pd.Timestamp.now().normalize()
    
    # Age vs DOB consistency
    if "dob" in df2.columns and "age" in df2.columns:
        dob_parsed = pd.to_datetime(df2["dob"], errors="coerce")
        computed_age = ((today - dob_parsed).dt.days // 365).replace({np.nan: None})
        mismatch = df2[(~dob_parsed.isna()) & (~df2["age"].isna()) & (abs(computed_age - df2["age"]) > 1)]
        if not mismatch.empty:
            issues["age_vs_dob"] = mismatch[["patient_id","dob","age"]].assign(computed_age=computed_age.loc[mismatch.index])
    
    # Admission/Discharge dates
    if "admission_date" in df2.columns and "discharge_date" in df2.columns:
        adm = pd.to_datetime(df2["admission_date"], errors="coerce")
        dis = pd.to_datetime(df2["discharge_date"], errors="coerce")
        bad_dates = df2[(~adm.isna()) & (~dis.isna()) & (adm > dis)]
        if not bad_dates.empty:
            issues["admission_after_discharge"] = bad_dates[["patient_id","admission_date","discharge_date"]]
    
    # Gender normalization / unexpected values
    if "gender" in df2.columns:
        allowed = {"M","F","Male","Female","O","Other","U","Unknown", None}
        unexpected = df2[~df2["gender"].isin(allowed) & ~df2["gender"].isna()]
        if not unexpected.empty:
            issues["unexpected_gender_codes"] = unexpected[["patient_id","gender"]]
    
    # Duplicate patient IDs
    if "patient_id" in df2.columns:
        dupes = df2[df2.duplicated(subset=["patient_id"], keep=False)].sort_values("patient_id")
        if not dupes.empty:
            issues["duplicate_patient_id"] = dupes
    
    return issues

def detect_potential_errors(df):
    issues = {}
    df2 = df.copy()
    ranges = {
        "heart_rate": (20, 250),
        "systolic_bp": (40, 300),
        "diastolic_bp": (20, 200),
        "temperature_c": (30.0, 45.0),
        "bmi": (8.0, 80.0)
    }
    for col, (low, high) in ranges.items():
        if col in df2.columns:
            bad = df2[(~df2[col].isna()) & ((df2[col] < low) | (df2[col] > high))]
            if not bad.empty:
                issues[f"out_of_range_{col}"] = bad[[c for c in ["patient_id",col] if c in bad.columns]]
    
    if "systolic_bp" in df2.columns and "diastolic_bp" in df2.columns:
        bad = df2[(~df2["systolic_bp"].isna()) & (~df2["diastolic_bp"].isna()) & (df2["systolic_bp"] <= df2["diastolic_bp"])]
        if not bad.empty:
            issues["systolic_not_greater_than_diastolic"] = bad[["patient_id","systolic_bp","diastolic_bp"]]
    
    if "hb_g_dl" in df2.columns:
        bad = df2[(~df2["hb_g_dl"].isna()) & ((df2["hb_g_dl"] < 4) | (df2["hb_g_dl"] > 25))]
        if not bad.empty:
            issues["hb_implausible"] = bad[["patient_id","hb_g_dl"]]
    
    return issues

def generate_summary_report(df):
    completeness = analyze_completeness(df).reset_index().rename(columns={"index":"field"})
    consistency = analyze_consistency(df)
    errors = detect_potential_errors(df)
    
    rows = []
    rows.append({"check":"total_records","detail":len(df)})
    
    for _, r in completeness.iterrows():
        rows.append({"check":"missing_summary", "detail": f"{r['field']}: {r['missing_count']} missing ({r['missing_percent']}%)"})
    
    for k,v in {**consistency, **errors}.items():
        rows.append({"check":"issue_detected", "detail": f"{k}: {len(v)} records"})
    
    report_df = pd.DataFrame(rows)
    
    out_dir = "/mnt/data/ehr_qc_output"
    os.makedirs(out_dir, exist_ok=True)
    report_csv = os.path.join(out_dir, "ehr_qc_summary.csv")
    report_df.to_csv(report_csv, index=False)
    
    for k,v in {**consistency, **errors}.items():
        fname = os.path.join(out_dir, f"{k}.csv")
        v.to_csv(fname, index=False)
    
    return report_df, out_dir

# Demo dataset
demo = pd.DataFrame([
    {"patient_id":"P001","dob":"1980-05-12","age":45,"gender":"F","admission_date":"2025-10-10","discharge_date":"2025-10-12",
     "heart_rate":78,"systolic_bp":120,"diastolic_bp":80,"temperature_c":37.0,"hb_g_dl":13.5,"bmi":24.5},
    {"patient_id":"P002","dob":"1990-01-01","age":35,"gender":"Male","admission_date":"2025-09-20","discharge_date":"2025-09-19",
     "heart_rate":10,"systolic_bp":80,"diastolic_bp":90,"temperature_c":36.8,"hb_g_dl":2.9,"bmi":22.0},
    {"patient_id":"P003","dob":"not a date","age":30,"gender":"X","admission_date":"2025-11-01","discharge_date":"2025-11-05",
     "heart_rate":85,"systolic_bp":110,"diastolic_bp":70,"temperature_c":40.5,"hb_g_dl":15.0,"bmi":300.0},
    {"patient_id":"P001","dob":"1980-05-12","age":45,"gender":"F","admission_date":"2025-10-10","discharge_date":"2025-10-12",
     "heart_rate":78,"systolic_bp":120,"diastolic_bp":80,"temperature_c":37.0,"hb_g_dl":13.5,"bmi":24.5},
    {"patient_id":"P004","dob":"2000-07-15","age":25,"gender":None,"admission_date":None,"discharge_date":None,
     "heart_rate":np.nan,"systolic_bp":np.nan,"diastolic_bp":np.nan,"temperature_c":np.nan,"hb_g_dl":np.nan,"bmi":np.nan},
])

comp = analyze_completeness(demo)
cons = analyze_consistency(demo)
errs = detect_potential_errors(demo)
report_df, out_dir = generate_summary_report(demo)

print("=== Completeness Summary ===")
print(comp)
print("\n=== Consistency Issues Found ===")
for k,v in cons.items():
    print(f"-- {k} ({len(v)} rows) --")
    display(v)
print("\n=== Potential Error Issues Found ===")
for k,v in errs.items():
    print(f"-- {k} ({len(v)} rows) --")
    display(v)

print(f"\nSaved detailed CSVs to: {out_dir}")
print(f"Summary file: {os.path.join(out_dir, 'ehr_qc_summary.csv')}")

demo.to_csv(os.path.join(out_dir, "demo_ehr_data.csv"), index=False)
os.path.exists(out_dir)

