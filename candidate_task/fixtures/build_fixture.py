"""Generate a small fictional bilingual PDF and labeled questions for smoke tests.

All statements below are invented test data, not DKU policy or service information.
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


PAGES = [
    ("Library hours", "The library closes at midnight on Friday.", "图书馆周五于午夜关闭。", "When does the library close on Friday?", "图书馆周五几点关闭？", "midnight"),
    ("Shuttle schedule", "The campus shuttle leaves at seven on Monday.", "校园巴士周一七点出发。", "When does the shuttle leave on Monday?", "校园巴士周一几点出发？", "seven"),
    ("Makerspace", "The makerspace workshop begins at six on Tuesday.", "创客空间的工作坊周二六点开始。", "When does the makerspace workshop begin?", "创客空间工作坊何时开始？", "six"),
    ("Writing studio", "The writing studio appointment is at two on Wednesday.", "写作中心预约安排在周三两点。", "When is the writing studio appointment?", "写作中心预约在什么时候？", "two"),
    ("Laboratory", "The laboratory safety briefing is at nine on Thursday.", "实验室安全说明会于周四九点举行。", "When is the laboratory safety briefing?", "实验室安全说明会什么时候举行？", "nine"),
    ("Archive", "The archive is closed on Sunday.", "档案馆周日关闭。", "When is the archive closed?", "档案馆哪天关闭？", "Sunday"),
    ("Tutoring", "The tutoring desk opens at ten on Saturday.", "辅导服务台周六十点开放。", "When does the tutoring desk open?", "辅导服务台什么时候开放？", "ten"),
    ("Room booking", "The seminar room must be booked two days in advance.", "研讨室必须提前两天预约。", "How far in advance must the seminar room be booked?", "研讨室需要提前几天预约？", "two days"),
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "demo_docs"
    destination.mkdir(exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(str(destination / "fictional_handbook.pdf"))
    cases = []
    for page, (title, english, chinese, en_question, zh_question, answer) in enumerate(PAGES, 1):
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(72, 750, f"Fictional handbook — {title}")
        pdf.setFont("Helvetica", 12)
        pdf.drawString(72, 705, english)
        pdf.setFont("STSong-Light", 12)
        pdf.drawString(72, 670, chinese)
        pdf.showPage()
        for language, question in (("en", en_question), ("zh", zh_question)):
            cases.append({
                "case_id": f"{language}-{page}",
                "language": language,
                "question": question,
                "expected_document": "fictional_handbook.pdf",
                "expected_page": page,
                "reference_answer": answer if language == "en" else chinese,
            })
    pdf.save()
    (root / "demo_questions.jsonl").write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8"
    )
    print(f"Created {len(PAGES)} fictional pages and {len(cases)} bilingual questions")


if __name__ == "__main__":
    main()
