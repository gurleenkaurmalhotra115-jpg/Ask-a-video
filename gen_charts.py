"""Generate vector PDF figures for IEEE paper: Duration Expansion and Tier Performance."""
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Group, Polygon
from reportlab.graphics import renderPDF

def create_duration_chart():
    # Width: 240 pt (single column width = 252 pt), Height: 110 pt
    w, h = 240, 110
    d = Drawing(w, h)
    
    # Background
    d.add(Rect(0, 0, w, h, fillColor=colors.HexColor("#f8fafc"), strokeColor=colors.HexColor("#cbd5e1"), strokeWidth=0.8, rx=4, ry=4))
    d.add(String(w/2, h - 12, "Search Space Expansion: Hit@1 vs Duration Bracket", fontName="Helvetica-Bold", fontSize=7, textAnchor="middle", fillColor=colors.HexColor("#0f172a")))
    
    # Axes
    ox, oy = 32, 22
    pw, ph = 195, 70
    d.add(Line(ox, oy, ox + pw, oy, strokeColor=colors.HexColor("#64748b"), strokeWidth=1))
    d.add(Line(ox, oy, ox, oy + ph, strokeColor=colors.HexColor("#64748b"), strokeWidth=1))
    
    # Y-axis ticks (0%, 10%, 20%, 30%)
    for pct, label in [(0, "0%"), (10, "10%"), (20, "20%"), (30, "30%")]:
        y = oy + (pct / 30.0) * ph
        d.add(Line(ox - 3, y, ox, y, strokeColor=colors.HexColor("#64748b"), strokeWidth=0.8))
        d.add(String(ox - 5, y - 2.5, label, fontName="Helvetica", fontSize=5.5, textAnchor="end", fillColor=colors.HexColor("#475569")))
        if pct > 0:
            d.add(Line(ox, y, ox + pw, y, strokeColor=colors.HexColor("#e2e8f0"), strokeWidth=0.5, strokeDashArray=[2,2]))

    # Data: Short (<=35m) vs Long (>35m)
    # fixed: 16.7% vs 6.0%
    # sentence: 10.0% vs 4.0%
    # semantic: 16.7% vs 3.0%
    # time_window: 23.3% vs 5.0%
    
    strats = [
        ("fixed_token", 16.7, 6.0, "#2563eb"),
        ("sentence", 10.0, 4.0, "#16a34a"),
        ("semantic", 16.7, 3.0, "#9333ea"),
        ("time_window", 23.3, 5.0, "#ea580c"),
    ]
    
    # Grouped bars: 2 groups (Short <=35m, Long >35m)
    group_x = [ox + 20, ox + 115]
    group_labels = ["Short (<=35 min, N=30)", "Long (>35 min, N=100)"]
    bar_w = 10
    
    for g_idx, gx in enumerate(group_x):
        d.add(String(gx + 2*bar_w + 3, oy - 9, group_labels[g_idx], fontName="Helvetica-Bold", fontSize=6, textAnchor="middle", fillColor=colors.HexColor("#1e293b")))
        for s_idx, (name, s_val, l_val, col) in enumerate(strats):
            val = s_val if g_idx == 0 else l_val
            bh = (val / 30.0) * ph
            bx = gx + s_idx * (bar_w + 2)
            d.add(Rect(bx, oy, bar_w, bh, fillColor=colors.HexColor(col), strokeColor=None))
            # Value label
            d.add(String(bx + bar_w/2, oy + bh + 1.5, f"{val:.0f}%", fontName="Helvetica", fontSize=4.5, textAnchor="middle", fillColor=colors.HexColor("#334155")))

    # Legend at top right
    leg_x = ox + 55
    leg_y = h - 22
    for s_idx, (name, _, _, col) in enumerate(strats):
        lx = leg_x + s_idx * 35
        d.add(Rect(lx, leg_y, 5, 5, fillColor=colors.HexColor(col), strokeColor=None))
        d.add(String(lx + 7, leg_y + 1, name.split("_")[0], fontName="Helvetica", fontSize=5, textAnchor="start", fillColor=colors.HexColor("#334155")))

    renderPDF.drawToFile(d, "fig_duration_expansion.pdf")
    print("fig_duration_expansion.pdf generated")

def create_tier_chart():
    # Width: 240 pt, Height: 110 pt
    w, h = 240, 110
    d = Drawing(w, h)
    
    # Background
    d.add(Rect(0, 0, w, h, fillColor=colors.HexColor("#f8fafc"), strokeColor=colors.HexColor("#cbd5e1"), strokeWidth=0.8, rx=4, ry=4))
    d.add(String(w/2, h - 12, "Pedagogical Tier Dynamics: Hit@1 and Hit@4 Recall", fontName="Helvetica-Bold", fontSize=7, textAnchor="middle", fillColor=colors.HexColor("#0f172a")))
    
    ox, oy = 32, 22
    pw, ph = 195, 70
    d.add(Line(ox, oy, ox + pw, oy, strokeColor=colors.HexColor("#64748b"), strokeWidth=1))
    d.add(Line(ox, oy, ox, oy + ph, strokeColor=colors.HexColor("#64748b"), strokeWidth=1))
    
    # Y-axis ticks (0%, 10%, 20%, 30%, 40%)
    for pct, label in [(0, "0%"), (10, "10%"), (20, "20%"), (30, "30%"), (40, "40%")]:
        y = oy + (pct / 40.0) * ph
        d.add(Line(ox - 3, y, ox, y, strokeColor=colors.HexColor("#64748b"), strokeWidth=0.8))
        d.add(String(ox - 5, y - 2.5, label, fontName="Helvetica", fontSize=5.5, textAnchor="end", fillColor=colors.HexColor("#475569")))
        if pct > 0:
            d.add(Line(ox, y, ox + pw, y, strokeColor=colors.HexColor("#e2e8f0"), strokeWidth=0.5, strokeDashArray=[2,2]))

    tiers = [
        ("Tier A (Studio, N=50)", 16.0, 34.0),
        ("Tier B (Classroom, N=50)", 4.0, 12.0),
        ("Tier C (NPTEL, N=30)", 3.3, 30.0),
    ]
    
    # Bar clusters per tier (Hit@1 vs Hit@4 for fixed_token baseline)
    bar_w = 14
    for t_idx, (t_name, h1, h4) in enumerate(tiers):
        cx = ox + 18 + t_idx * 62
        d.add(String(cx + bar_w, oy - 9, t_name.split(" ")[0] + " " + t_name.split(" ")[1], fontName="Helvetica-Bold", fontSize=5.5, textAnchor="middle", fillColor=colors.HexColor("#1e293b")))
        
        # Hit@1 bar
        bh1 = (h1 / 40.0) * ph
        d.add(Rect(cx, oy, bar_w, bh1, fillColor=colors.HexColor("#2563eb"), strokeColor=None))
        d.add(String(cx + bar_w/2, oy + bh1 + 1.5, f"{h1:.1f}%", fontName="Helvetica", fontSize=5, textAnchor="middle", fillColor=colors.HexColor("#1e3a8a")))
        
        # Hit@4 bar
        bh4 = (h4 / 40.0) * ph
        d.add(Rect(cx + bar_w + 2, oy, bar_w, bh4, fillColor=colors.HexColor("#93c5fd"), strokeColor=None))
        d.add(String(cx + bar_w + 2 + bar_w/2, oy + bh4 + 1.5, f"{h4:.1f}%", fontName="Helvetica", fontSize=5, textAnchor="middle", fillColor=colors.HexColor("#1e3a8a")))

    # Legend
    leg_x = ox + 80
    leg_y = h - 22
    d.add(Rect(leg_x, leg_y, 7, 5, fillColor=colors.HexColor("#2563eb"), strokeColor=None))
    d.add(String(leg_x + 9, leg_y + 1, "Hit@1 (Top-1 Precision)", fontName="Helvetica", fontSize=5.5, textAnchor="start", fillColor=colors.HexColor("#334155")))
    d.add(Rect(leg_x + 65, leg_y, 7, 5, fillColor=colors.HexColor("#93c5fd"), strokeColor=None))
    d.add(String(leg_x + 74, leg_y + 1, "Hit@4 (Top-4 Candidate Recall)", fontName="Helvetica", fontSize=5.5, textAnchor="start", fillColor=colors.HexColor("#334155")))

    renderPDF.drawToFile(d, "fig_tier_performance.pdf")
    print("fig_tier_performance.pdf generated")

if __name__ == "__main__":
    create_duration_chart()
    create_tier_chart()
