"""Generate high-resolution vector PDF architecture diagram for IEEE paper."""
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.graphics import renderPDF

def create_diagram():
    # Width: 504 pt (standard IEEE textwidth = 7 inches * 72 = 504 pt), Height: 115 pt
    w, h = 504, 115
    d = Drawing(w, h)
    
    # Background card
    d.add(Rect(0, 0, w, h, fillColor=colors.HexColor("#f8fafc"), strokeColor=colors.HexColor("#cbd5e1"), strokeWidth=1, rx=6, ry=6))
    
    # Pipeline stages (Top row: Ingestion & Indexing)
    boxes_top = [
        ("Video Source\n(YouTube / Whisper)", 10, 62, 78, 42, "#eff6ff", "#2563eb"),
        ("ASR Normalization\n(Punctuation Restore)", 104, 62, 85, 42, "#f0fdf4", "#16a34a"),
        ("Pluggable Chunking\n(4 Strategies)", 205, 62, 85, 42, "#faf5ff", "#9333ea"),
        ("Dense Bi-Encoder\n(all-MiniLM-L6-v2)", 306, 62, 85, 42, "#fff7ed", "#ea580c"),
        ("Local Disk Qdrant\n(path='qdrant_data')", 407, 62, 87, 42, "#fef2f2", "#dc2626"),
    ]
    
    for text, bx, by, bw, bh, fcol, scol in boxes_top:
        d.add(Rect(bx, by, bw, bh, fillColor=colors.HexColor(fcol), strokeColor=colors.HexColor(scol), strokeWidth=1.2, rx=4, ry=4))
        lines = text.split("\n")
        if len(lines) == 1:
            d.add(String(bx + bw/2, by + 16, lines[0], fontName="Helvetica-Bold", fontSize=7.5, textAnchor="middle", fillColor=colors.HexColor("#0f172a")))
        else:
            d.add(String(bx + bw/2, by + 23, lines[0], fontName="Helvetica-Bold", fontSize=7.5, textAnchor="middle", fillColor=colors.HexColor("#0f172a")))
            d.add(String(bx + bw/2, by + 11, lines[1], fontName="Helvetica", fontSize=6.5, textAnchor="middle", fillColor=colors.HexColor("#475569")))

    # Connectors for top row
    for i in range(len(boxes_top) - 1):
        x1 = boxes_top[i][1] + boxes_top[i][3]
        x2 = boxes_top[i+1][1]
        y = 83
        d.add(Line(x1, y, x2, y, strokeColor=colors.HexColor("#64748b"), strokeWidth=1.5))
        d.add(Polygon(points=[x2, y, x2-5, y+3, x2-5, y-3], fillColor=colors.HexColor("#64748b"), strokeColor=colors.HexColor("#64748b")))

    # Bottom row: Query & Grounded Generation
    boxes_bot = [
        ("Student Query q\n(Evaluation Probe)", 10, 10, 78, 40, "#eff6ff", "#2563eb"),
        ("Top-k Temporal\nRetriever", 104, 10, 85, 40, "#f0fdf4", "#16a34a"),
        ("Closed-World Prompt\n([n] Attribution)", 205, 10, 85, 40, "#faf5ff", "#9333ea"),
        ("Gemini 3.5 Flash Lite\n(Grounding Guardrail)", 306, 10, 85, 40, "#fff7ed", "#ea580c"),
        ("Interactive Player\n(Seek to mm:ss)", 407, 10, 87, 40, "#ecfdf5", "#059669"),
    ]

    for text, bx, by, bw, bh, fcol, scol in boxes_bot:
        d.add(Rect(bx, by, bw, bh, fillColor=colors.HexColor(fcol), strokeColor=colors.HexColor(scol), strokeWidth=1.2, rx=4, ry=4))
        lines = text.split("\n")
        if len(lines) == 1:
            d.add(String(bx + bw/2, by + 16, lines[0], fontName="Helvetica-Bold", fontSize=7.5, textAnchor="middle", fillColor=colors.HexColor("#0f172a")))
        else:
            d.add(String(bx + bw/2, by + 22, lines[0], fontName="Helvetica-Bold", fontSize=7.5, textAnchor="middle", fillColor=colors.HexColor("#0f172a")))
            d.add(String(bx + bw/2, by + 10, lines[1], fontName="Helvetica", fontSize=6.5, textAnchor="middle", fillColor=colors.HexColor("#475569")))

    # Connectors for bottom row
    for i in range(len(boxes_bot) - 1):
        x1 = boxes_bot[i][1] + boxes_bot[i][3]
        x2 = boxes_bot[i+1][1]
        y = 30
        d.add(Line(x1, y, x2, y, strokeColor=colors.HexColor("#64748b"), strokeWidth=1.5))
        d.add(Polygon(points=[x2, y, x2-5, y+3, x2-5, y-3], fillColor=colors.HexColor("#64748b"), strokeColor=colors.HexColor("#64748b")))

    # Vertical vector-search feed line (from Qdrant to Top-k Retriever)
    d.add(Line(450, 62, 450, 53, strokeColor=colors.HexColor("#2563eb"), strokeWidth=1.2, strokeDashArray=[2,2]))
    d.add(Line(450, 53, 146, 53, strokeColor=colors.HexColor("#2563eb"), strokeWidth=1.2, strokeDashArray=[2,2]))
    d.add(Line(146, 53, 146, 50, strokeColor=colors.HexColor("#2563eb"), strokeWidth=1.2, strokeDashArray=[2,2]))
    d.add(Polygon(points=[146, 50, 143, 54, 149, 54], fillColor=colors.HexColor("#2563eb"), strokeColor=colors.HexColor("#2563eb")))
    d.add(String(298, 54.5, "Cosine Nearest Neighbors", fontName="Helvetica-Oblique", fontSize=6, textAnchor="middle", fillColor=colors.HexColor("#2563eb")))

    renderPDF.drawToFile(d, "fig_architecture.pdf")
    print("fig_architecture.pdf generated successfully")

if __name__ == "__main__":
    create_diagram()
