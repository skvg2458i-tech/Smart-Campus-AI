"""PDF fine receipts (ReportLab). Every receipt is watermarked DEMONSTRATION ONLY."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from config import APP_NAME, RECEIPTS_DIR, ensure_dirs

NAVY = colors.HexColor("#0F2A4A")
TEAL = colors.HexColor("#0E9F8E")
GREY = colors.HexColor("#5B6573")


def generate_receipt(fine: dict, out_dir: Path | None = None) -> Path:
    ensure_dirs()
    out_dir = Path(out_dir or RECEIPTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{fine['receipt_no']}.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    W, H = A4
    c.setTitle(f"DEMO receipt {fine['receipt_no']}")

    # watermark
    c.saveState()
    c.setFillColor(colors.Color(0.85, 0.1, 0.1, alpha=0.12))
    c.setFont("Helvetica-Bold", 54)
    c.translate(W / 2, H / 2)
    c.rotate(35)
    c.drawCentredString(0, 0, "DEMONSTRATION ONLY")
    c.restoreState()

    # header band
    c.setFillColor(NAVY)
    c.rect(0, H - 38 * mm, W, 38 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(18 * mm, H - 17 * mm, APP_NAME)
    c.setFont("Helvetica", 10)
    c.drawString(18 * mm, H - 24 * mm, "Campus Cleanliness - Demonstration Fine Receipt (AIML academic project)")
    c.setFillColor(TEAL)
    c.rect(0, H - 40 * mm, W, 2 * mm, stroke=0, fill=1)

    # demo notice
    c.setFillColor(colors.HexColor("#B42318"))
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(W / 2, H - 50 * mm, "DEMONSTRATION ONLY - NOT A REAL FINE - NO PAYMENT IS DUE")
    c.setFont("Helvetica", 8.5)
    c.drawCentredString(W / 2, H - 55 * mm,
                        "Generated for a college project using fictional student records. Not valid for any purpose.")

    def row(y, k, v):
        c.setFillColor(GREY); c.setFont("Helvetica", 10); c.drawString(22 * mm, y, k)
        c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 10); c.drawString(75 * mm, y, str(v))

    y = H - 70 * mm
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 12); c.drawString(18 * mm, y, "Receipt"); y -= 8 * mm
    for k, v in [("Receipt number", fine["receipt_no"]), ("Issued at", fine["issued_at"]),
                 ("Issued by (admin)", fine["issued_by"]),
                 ("Status", "PAID (SIMULATED)" if fine["status"] == "PAID_SIMULATED" else "UNPAID (demo)")]:
        row(y, k, v); y -= 6.5 * mm
    if fine.get("paid_at"):
        row(y, "Simulated payment at", fine["paid_at"]); y -= 6.5 * mm

    y -= 4 * mm
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 12); c.drawString(18 * mm, y, "Fictional student"); y -= 8 * mm
    for k, v in [("Name", fine["student_name"]), ("USN (fictional)", fine["usn"]),
                 ("Department", fine["department"]), ("Semester", fine["semester"])]:
        row(y, k, v); y -= 6.5 * mm

    y -= 4 * mm
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 12); c.drawString(18 * mm, y, "Incident"); y -= 8 * mm
    for k, v in [("Incident ID", fine["incident_id"]), ("Event time", fine["event_time"]),
                 ("AI category", fine["detection_category"]), ("Object", fine["object_category"]),
                 ("AI confidence", f"{fine['confidence']:.2f}"),
                 ("Verified by", f"{fine['reviewed_by']} at {fine['reviewed_at']}")]:
        row(y, k, v); y -= 6.5 * mm

    y -= 6 * mm
    c.setFillColor(colors.HexColor("#F1F5F9"))
    c.roundRect(18 * mm, y - 16 * mm, W - 36 * mm, 18 * mm, 3 * mm, stroke=0, fill=1)
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 14)
    c.drawString(24 * mm, y - 10 * mm, "Demonstration amount")
    c.drawRightString(W - 24 * mm, y - 10 * mm, f"{fine['currency']} {fine['amount']:,.2f}")

    c.setFillColor(GREY); c.setFont("Helvetica", 8)
    notes = [
        "This incident was flagged by an AI system and verified by an administrator before this demo receipt was created.",
        "AI detections and track-based person associations can be wrong; no facial recognition was used.",
        "Evidence is stored only on the local demonstration machine.",
    ]
    for i, n in enumerate(notes):
        c.drawString(18 * mm, 22 * mm - i * 4.5 * mm, n)
    c.setFillColor(colors.HexColor("#B42318")); c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W / 2, 8 * mm, "DEMONSTRATION ONLY")
    c.showPage()
    c.save()
    return path
