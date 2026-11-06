"""Generate complete PDF solution for Computational Methods Assignment (ES-201)."""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_FILENAME = "Computational_Methods_Assignment_Solutions.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 800, "Maharaja Surajmal Institute of Technology | ES-201: Computational Methods")
            self.drawRightString(541, 800, "Student Assignment Solutions")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 794, 541, 794)

        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawString(54, 35, "B.Tech. Third Semester | Units I & II Complete Solutions")
        self.drawRightString(541, 35, page_str)
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 45, 541, 45)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_FILENAME,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=1, # Center
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        alignment=1,
        textColor=colors.HexColor("#2B6CB0"),
        spaceAfter=12
    )

    unit_header_style = ParagraphStyle(
        'UnitHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    q_title_style = ParagraphStyle(
        'QuestionTitle',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0D3880"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    subpart_style = ParagraphStyle(
        'SubpartTitle',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#2C5282"),
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=body_style,
        fontName='Courier',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1A202C")
    )

    box_style = ParagraphStyle(
        'BoxStyle',
        parent=body_style,
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#0F4C81")
    )

    table_cell = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        alignment=1
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        alignment=1
    )

    story = []

    # Title block
    story.append(Paragraph("MAHARAJA SURAJMAL INSTITUTE OF TECHNOLOGY", title_style))
    story.append(Paragraph("<b>DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING / IT / ECE</b><br/>ES-201: COMPUTATIONAL METHODS — STUDENT ASSIGNMENT SOLUTIONS<br/><b>Units I & II | B.Tech. Third Semester</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A365D"), spaceBefore=0, spaceAfter=10))

    # =========================================================================
    # UNIT I
    # =========================================================================
    story.append(Paragraph("UNIT I | Roots of Equations and Optimization", unit_header_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#2B6CB0"), spaceBefore=0, spaceAfter=8))

    # --- Q1 ---
    story.append(Paragraph("Q1. Taylor series, Rolle’s theorem and the Mean Value Theorem", q_title_style))
    
    story.append(Paragraph("<b>(a) Expand eˣ in a Maclaurin series up to x⁴, approximate e⁰·¹, and write the remainder term.</b>", subpart_style))
    story.append(Paragraph(
        "<b>Maclaurin Series Formula:</b><br/>"
        "&nbsp;&nbsp;f(x) = f(0) + x·f'(0) + (x²/2!)·f''(0) + (x³/3!)·f'''(0) + (x⁴/4!)·f⁽⁴⁾(0) + R₄(x)<br/>"
        "For f(x) = eˣ, f⁽ᵏ⁾(0) = 1 for all k ≥ 0.<br/>"
        "&nbsp;&nbsp;<b>eˣ = 1 + x + x²/2 + x³/6 + x⁴/24 + R₄(x)</b><br/><br/>"
        "<b>Approximation for x = 0.1:</b><br/>"
        "&nbsp;&nbsp;e⁰·¹ ≈ 1 + 0.1 + (0.01)/2 + (0.001)/6 + (0.0001)/24<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= 1 + 0.1 + 0.005 + 0.00016667 + 0.00000417 = <b>1.10517084</b><br/><br/>"
        "<b>Lagrange Remainder Term:</b><br/>"
        "&nbsp;&nbsp;R₄(x) = [x⁵ / 5!] · f⁽⁵⁾(ξ) = <b>(x⁵ / 120) · e^ξ</b>, for some ξ ∈ (0, x).<br/>"
        "For x = 0.1: R₄(0.1) = (10⁻⁵ / 120)·e^ξ. Since 0 &lt; ξ &lt; 0.1 and e^ξ &lt; e⁰·¹ &lt; 1.1052:<br/>"
        "&nbsp;&nbsp;<b>0 &lt; R₄(0.1) &lt; (1.1052 / 120) × 10⁻⁵ ≈ 9.21 × 10⁻⁸</b>",
        body_style
    ))

    story.append(Paragraph("<b>(b) Verify Rolle’s theorem for f(x) = −x² + 8x on [2, 6].</b>", subpart_style))
    story.append(Paragraph(
        "<b>Statement:</b> If f(x) is (i) continuous on [a, b], (ii) differentiable on (a, b), and (iii) f(a) = f(b), then ∃ c ∈ (a, b) such that f'(c) = 0.<br/>"
        "<b>Verification:</b><br/>"
        "1. f(x) = −x² + 8x is a polynomial, hence continuous on [2, 6].<br/>"
        "2. f'(x) = −2x + 8 exists for all x ∈ (2, 6), so f(x) is differentiable on (2, 6).<br/>"
        "3. Boundary values: f(2) = −(2)² + 8(2) = 12, and f(6) = −(6)² + 8(6) = 12. Thus f(2) = f(6) = 12.<br/>"
        "4. Point c: f'(c) = −2c + 8 = 0 ⇒ <b>c = 4</b>.<br/>"
        "Since c = 4 lies strictly in (2, 6), <b>Rolle's Theorem is verified.</b>",
        body_style
    ))

    story.append(Paragraph("<b>(c) State Lagrange’s Mean Value Theorem (LMVT) and verify it for f(x) = x³/3 + 2x on [0, 3].</b>", subpart_style))
    story.append(Paragraph(
        "<b>Statement:</b> If f(x) is continuous on [a, b] and differentiable on (a, b), then ∃ c ∈ (a, b) such that f'(c) = [f(b) − f(a)] / (b − a).<br/>"
        "<b>Verification:</b><br/>"
        "1. f(x) is a polynomial, continuous on [0, 3] and differentiable on (0, 3).<br/>"
        "2. f'(x) = x² + 2.<br/>"
        "3. f(0) = 0, f(3) = (3³)/3 + 2(3) = 9 + 6 = 15.<br/>"
        "4. By LMVT: f'(c) = [15 − 0] / [3 − 0] = 5 ⇒ c² + 2 = 5 ⇒ c² = 3 ⇒ <b>c = √3 ≈ 1.732 ∈ (0, 3)</b>.<br/>"
        "Hence, <b>Lagrange's MVT is verified.</b>",
        body_style
    ))

    # --- Q2 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q2. Approximation, errors, floating-point arithmetic and loss of significance", q_title_style))
    story.append(Paragraph("<b>(a) Round 37.46235 to four significant figures; determine errors; rounding vs truncation.</b>", subpart_style))
    story.append(Paragraph(
        "• True value x = 37.46235. The first 4 significant figures are 3, 7, 4, 6. The 5th digit is 2 (&lt; 5).<br/>"
        "&nbsp;&nbsp;Rounded value x* = <b>37.46</b>.<br/>"
        "• <b>Absolute Error (Eₐ):</b> |x − x*| = |37.46235 − 37.46| = <b>0.00235</b>.<br/>"
        "• <b>Relative Error (Eᵣ):</b> Eₐ / |x| = 0.00235 / 37.46235 ≈ <b>6.273 × 10⁻⁵</b>.<br/>"
        "• <b>Percentage Error (Eₚ):</b> Eᵣ × 100% ≈ <b>0.006273%</b>.<br/>"
        "• <b>Rounding vs Truncation:</b> Rounding adjusts the last kept digit based on whether the first dropped digit is ≥ 5 (maximum error ≤ ½·β¹⁻ᵏ). Truncation simply discards all trailing digits without adjustment (maximum error ≤ β¹⁻ᵏ).",
        body_style
    ))

    story.append(Paragraph("<b>(b) Represent 752.6835 in normalized decimal floating-point (4-digit mantissa) and find round-off error.</b>", subpart_style))
    story.append(Paragraph(
        "• Normalized scientific form: 752.6835 = 0.7526835 × 10³.<br/>"
        "• With a 4-digit mantissa, the 5th digit is 8 (≥ 5), so the 4th digit rounds up: <b>fl(x) = 0.7527 × 10³ = 752.7</b>.<br/>"
        "• <b>Round-off Error:</b> |752.6835 − 752.7| = <b>0.0165</b> (Relative round-off error ≈ 2.192 × 10⁻⁵).<br/>"
        "• <b>Effects of Finite-Precision:</b> Causes catastrophic cancellation when subtracting nearly equal numbers, breakdown of associativity [(a + b) + c ≠ a + (b + c)], and underflow/overflow in iterative schemes.",
        body_style
    ))

    story.append(Paragraph("<b>(c) Compute 1 − cos(0.1) directly vs 2sin²(0.05). Explain numerical reliability.</b>", subpart_style))
    story.append(Paragraph(
        "• <b>Direct computation:</b> cos(0.1) ≈ 0.9950041653 ⇒ 1 − cos(0.1) = <b>0.0049958347</b>.<br/>"
        "• <b>Using 2sin²(0.05):</b> sin(0.05) ≈ 0.0499791693 ⇒ 2·(0.0499791693)² = <b>0.0049958347</b>.<br/>"
        "• <b>Numerical Reliability:</b> <b>2sin²(0.05) is far more reliable</b>. As x → 0, cos(x) ≈ 1. Direct computation involves subtracting two nearly equal numbers (1 and 0.995...), which causes <i>catastrophic subtractive cancellation</i>, severely destroying significant precision bits in finite-precision registers. The half-angle identity replaces subtraction with stable multiplication.",
        body_style
    ))

    # --- Q3 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q3. Bisection method: algorithm and convergence", q_title_style))
    story.append(Paragraph(
        "Find real root of f(x) = x³ − 4x − 9 = 0 in [2, 3] (6 iterations).<br/>"
        "<b>Convergence Justification:</b> f(x) is continuous. f(2) = 2³ − 4(2) − 9 = −9 &lt; 0; f(3) = 3³ − 4(3) − 9 = +6 &gt; 0. Since f(2)·f(3) &lt; 0, by Intermediate Value Theorem, a real root exists in [2, 3].<br/>"
        "<b>Algorithm:</b> At step n, cₙ = (aₙ + bₙ)/2. If f(aₙ)·f(cₙ) &lt; 0, set [aₙ₊₁, bₙ₊₁] = [aₙ, cₙ]; else set [aₙ₊₁, bₙ₊₁] = [cₙ, bₙ].",
        body_style
    ))

    # Bisection Table
    bis_data = [
        [Paragraph("<b>n</b>", table_cell_bold), Paragraph("<b>aₙ</b>", table_cell_bold), Paragraph("<b>bₙ</b>", table_cell_bold), Paragraph("<b>cₙ = (aₙ+bₙ)/2</b>", table_cell_bold), Paragraph("<b>f(cₙ)</b>", table_cell_bold), Paragraph("<b>Sign</b>", table_cell_bold), Paragraph("<b>Updated Bracket</b>", table_cell_bold)],
        ["1", "2.00000", "3.00000", "2.50000", "−3.37500", "−", "[2.50000, 3.00000]"],
        ["2", "2.50000", "3.00000", "2.75000", "+0.79688", "+", "[2.50000, 2.75000]"],
        ["3", "2.50000", "2.75000", "2.62500", "−1.41211", "−", "[2.62500, 2.75000]"],
        ["4", "2.62500", "2.75000", "2.68750", "−0.33911", "−", "[2.68750, 2.75000]"],
        ["5", "2.68750", "2.75000", "2.71875", "+0.22079", "+", "[2.68750, 2.71875]"],
        ["6", "2.68750", "2.71875", "2.70313", "−0.06114", "−", "[2.70313, 2.71875]"]
    ]
    t_bis = Table(bis_data, colWidths=[25, 65, 65, 85, 70, 45, 130])
    t_bis.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_bis)
    story.append(Paragraph(
        "<b>Root after 6 iterations:</b> x ≈ <b>2.70313</b> (or midpoint x ≈ <b>2.71094</b>).<br/>"
        "<b>Theoretical Error Bound:</b> |x* − c₆| ≤ (b − a)/2⁶ = (3 − 2)/64 = 1/64 = <b>0.015625</b>.",
        body_style
    ))

    # --- Q4 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q4. Newton–Raphson method: implementation and convergence", q_title_style))
    story.append(Paragraph(
        "Evaluate ∛24 with x₀ = 3.<br/>"
        "• Let x = ∛24 ⇒ f(x) = x³ − 24 = 0, and f'(x) = 3x².<br/>"
        "• <b>Iteration Formula:</b> xₙ₊₁ = xₙ − (xₙ³ − 24)/(3xₙ²) = <b>(2xₙ³ + 24) / (3xₙ²) = (2/3)xₙ + 8/xₙ²</b>.<br/>"
        "<b>Iterations:</b><br/>"
        "&nbsp;&nbsp;n = 0: x₁ = [2(3)³ + 24] / [3(3)²] = 78 / 27 ≈ <b>2.888889</b><br/>"
        "&nbsp;&nbsp;n = 1: x₂ = [2(2.888889)³ + 24] / [3(2.888889)²] = 72.197531 / 25.037037 ≈ <b>2.884504</b><br/>"
        "&nbsp;&nbsp;n = 2: x₃ = [2(2.884504)³ + 24] / [3(2.884504)²] ≈ <b>2.884499</b><br/>"
        "Comparing x₂ and x₃: both round to <b>2.8845</b> (agreed to 4 decimal places).<br/>"
        "<b>Conditions for Local Quadratic Convergence:</b> (i) f(α) = 0, (ii) f'(α) ≠ 0 (simple root), (iii) f''(x) is continuous near α, (iv) x₀ sufficiently close to α.<br/>"
        "<b>Proof:</b> Let eₙ = xₙ − α. Taylor expansion of f(α) about xₙ: 0 = f(xₙ) − eₙ·f'(xₙ) + (eₙ²/2)·f''(ξₙ).<br/>"
        "Dividing by f'(xₙ) and substituting xₙ₊₁ − α = eₙ₊₁ = eₙ − f(xₙ)/f'(xₙ) gives: eₙ₊₁ = [f''(ξₙ)/(2f'(xₙ))]·eₙ².<br/>"
        "Taking limits: lim(n→∞) |eₙ₊₁| / |eₙ|² = |f''(α) / [2f'(α)]| = C (constant). Hence convergence order is <b>2 (quadratic)</b>.",
        body_style
    ))

    # --- Q5 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q5. Secant method: algorithm and order of convergence", q_title_style))
    story.append(Paragraph(
        "Find root of f(x) = x³ − 2x − 5 = 0 starting with x₀ = 2, x₁ = 3.<br/>"
        "• <b>Formula:</b> xₙ₊₁ = xₙ − f(xₙ)·[(xₙ − xₙ₋₁) / (f(xₙ) − f(xₙ₋₁))].<br/>"
        "f(x₀) = f(2) = 2³ − 2(2) − 5 = −1; f(x₁) = f(3) = 27 − 6 − 5 = 16.",
        body_style
    ))
    sec_data = [
        [Paragraph("<b>n</b>", table_cell_bold), Paragraph("<b>xₙ₋₁</b>", table_cell_bold), Paragraph("<b>xₙ</b>", table_cell_bold), Paragraph("<b>f(xₙ₋₁)</b>", table_cell_bold), Paragraph("<b>f(xₙ)</b>", table_cell_bold), Paragraph("<b>xₙ₊₁</b>", table_cell_bold)],
        ["1", "2.00000", "3.00000", "−1.00000", "16.00000", "2.05882"],
        ["2", "3.00000", "2.05882", "16.00000", "−0.39080", "2.08126"],
        ["3", "2.05882", "2.08126", "−0.39080", "−0.14720", "2.09482"],
        ["4", "2.08126", "2.09482", "−0.14720", "+0.00326", "2.09455"]
    ]
    t_sec = Table(sec_data, colWidths=[30, 80, 80, 80, 80, 85])
    t_sec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_sec)
    story.append(Paragraph(
        "Correct to three decimal places: <b>x ≈ 2.095</b>.<br/>"
        "• <b>Order Comparison:</b> Bisection: p = 1 (linear, slowest, robust). Secant: p = (1+√5)/2 ≈ <b>1.618</b> (superlinear, no derivative evaluations required). Newton–Raphson: p = <b>2</b> (quadratic, requires derivative at each step).",
        body_style
    ))

    # --- Q6 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q6. One-variable minimization: Fibonacci search", q_title_style))
    story.append(Paragraph(
        "Minimize f(x) = x² + 54/x over [1, 5] using F₅.<br/>"
        "• Fibonacci sequence: F₀=1, F₁=1, F₂=2, F₃=3, F₄=5, F₅=8.<br/>"
        "• <b>Why x = 0 is excluded:</b> The term 54/x has a singularity at x = 0 (division by zero; f(x) → ∞ as x → 0⁺). Unimodality and continuity are violated across 0.<br/>"
        "<b>Iterations:</b><br/>"
        "1. [a₁, b₁] = [1, 5], L₁ = 4. Ratio = F₃/F₅ = 3/8.<br/>"
        "&nbsp;&nbsp;x₁ = 1 + (3/8)(4) = 2.5 ⇒ f(2.5) = (2.5)² + 54/2.5 = 6.25 + 21.6 = 27.85<br/>"
        "&nbsp;&nbsp;x₂ = 5 − (3/8)(4) = 3.5 ⇒ f(3.5) = (3.5)² + 54/3.5 = 12.25 + 15.43 = 27.68<br/>"
        "&nbsp;&nbsp;Since f(x₁) &gt; f(x₂), discard [1, 2.5]. <b>Updated Bracket: [2.5, 5.0]</b>.<br/>"
        "2. [a₂, b₂] = [2.5, 5.0], L₂ = 2.5. Reused point x₁' = 3.5, f(3.5) = 27.68. Ratio = F₂/F₄ = 2/5.<br/>"
        "&nbsp;&nbsp;x₂' = 2.5 + (3/5)(2.5) = 4.0 ⇒ f(4.0) = 16 + 13.5 = 29.5. Since f(3.5) &lt; f(4.0), discard [4.0, 5.0]. <b>Updated Bracket: [2.5, 4.0]</b>.<br/>"
        "3. [a₃, b₃] = [2.5, 4.0], L₃ = 1.5. Reused point x₂'' = 3.5. Ratio = F₁/F₃ = 1/3.<br/>"
        "&nbsp;&nbsp;x₁'' = 2.5 + (1/3)(1.5) = 3.0 ⇒ f(3.0) = 3² + 54/3 = 9 + 18 = 27.0.<br/>"
        "&nbsp;&nbsp;Since f(3.0) &lt; f(3.5), discard [3.5, 4.0]. <b>Final Bracket: [2.5, 3.5]</b>, with minimum at x = <b>3.0</b>.",
        body_style
    ))

    # --- Q7 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q7. One-variable minimization: golden-section search", q_title_style))
    story.append(Paragraph(
        "Minimize f(x) = x⁴ − 14x³ + 60x² − 70x on [0, 2] (4 iterations). Golden ratio φ = (√5 − 1)/2 ≈ 0.618034.<br/>"
        "Interior points: x₁ = a + (1 − φ)(b − a) = a + 0.381966·L; x₂ = a + φ·L = a + 0.618034·L.<br/>"
        "<b>Iteration 1:</b> [0, 2], L = 2 ⇒ x₁ = 0.763932, x₂ = 1.236068.<br/>"
        "&nbsp;&nbsp;f(x₁) = −24.363, f(x₂) = −21.037. Since f(x₁) &lt; f(x₂), discard [x₂, 2]. <b>Bracket: [0, 1.236068]</b>.<br/>"
        "<b>Iteration 2:</b> [0, 1.236068], L = 1.236068. Reused x₂ = 0.763932, f(x₂) = −24.363.<br/>"
        "&nbsp;&nbsp;x₁ = 0 + 0.381966(1.236068) = 0.472136 ⇒ f(x₁) = −21.037. Since f(x₁) &gt; f(x₂), discard [0, x₁]. <b>Bracket: [0.472136, 1.236068]</b>.<br/>"
        "<b>Iteration 3:</b> [0.472136, 1.236068], L = 0.763932. Reused x₁ = 0.763932, f(x₁) = −24.363.<br/>"
        "&nbsp;&nbsp;x₂ = 0.472136 + 0.618034(0.763932) = 0.944272 ⇒ f(x₂) = −24.363. Discard [0.944272, 1.236068]. <b>Bracket: [0.472136, 0.944272]</b>.<br/>"
        "<b>Iteration 4:</b> [0.472136, 0.944272], L = 0.472136. Reused x₂ = 0.763932, f = −24.363.<br/>"
        "&nbsp;&nbsp;x₁ = 0.472136 + 0.381966(0.472136) = 0.652476 ⇒ f(x₁) = −23.82. Since f(x₁) &gt; f(x₂), discard [0.472136, x₁].<br/>"
        "&nbsp;&nbsp;<b>Final Updated Interval: [0.652476, 0.944272]</b>.",
        body_style
    ))

    # --- Q8 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q8. One-variable minimization: Newton’s method", q_title_style))
    story.append(Paragraph(
        "Find local min of f(x) = 0.25x⁴ − x² − 5x + 1 in [0, 3], starting at x₀ = 2, stop when |xₖ₊₁ − xₖ| &lt; 0.01.<br/>"
        "• Derivatives: f'(x) = x³ − 2x − 5, and f''(x) = 3x² − 2.<br/>"
        "• Newton's minimization formula: xₖ₊₁ = xₖ − f'(xₖ) / f''(xₖ).<br/>"
        "<b>Iteration 0:</b> x₀ = 2 ⇒ f'(2) = 2³ − 2(2) − 5 = −1; f''(2) = 3(4) − 2 = 10.<br/>"
        "&nbsp;&nbsp;x₁ = 2 − (−1 / 10) = <b>2.1000</b>. Step |x₁ − x₀| = 0.10 ≥ 0.01.<br/>"
        "<b>Iteration 1:</b> x₁ = 2.1 ⇒ f'(2.1) = (2.1)³ − 2(2.1) − 5 = 9.261 − 4.2 − 5 = 0.061.<br/>"
        "&nbsp;&nbsp;f''(2.1) = 3(2.1)² − 2 = 13.23 − 2 = 11.23.<br/>"
        "&nbsp;&nbsp;x₂ = 2.1 − (0.061 / 11.23) = 2.1 − 0.005432 = <b>2.094568</b>.<br/>"
        "&nbsp;&nbsp;Step |x₂ − x₁| = |2.094568 − 2.1000| = <b>0.005432 &lt; 0.01</b>.<br/>"
        "Stopping condition met. <b>Local Minimum at x ≈ 2.0946</b>.<br/>"
        "<b>Second-Order Condition:</b> f''(2.0946) = 3(2.0946)² − 2 = 11.162 &gt; 0. Since f'(x*) = 0 and f''(x*) &gt; 0, it is strictly a <b>local minimum</b>.",
        body_style
    ))

    # --- Q9 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q9. Multivariable minimization: steepest descent", q_title_style))
    story.append(Paragraph(
        "Minimize f(x₁, x₂) = x₁ − x₂ + 2x₁² + 2x₁x₂ + x₂² starting from X₀ = (0, 0)ᵀ (3 iterations with exact line search).<br/>"
        "• Gradient: ∇f(X) = [1 + 4x₁ + 2x₂, −1 + 2x₁ + 2x₂]ᵀ; Hessian H = [[4, 2], [2, 2]].<br/>"
        "• Exact step length: αₖ = [dₖᵀ dₖ] / [dₖᵀ H dₖ] along dₖ = −∇f(Xₖ).<br/>"
        "<b>Iteration 1:</b> X₀ = (0, 0)ᵀ ⇒ ∇f(X₀) = [1, −1]ᵀ ⇒ d₀ = [−1, 1]ᵀ.<br/>"
        "&nbsp;&nbsp;d₀ᵀ d₀ = 2; H d₀ = [−2, 0]ᵀ; d₀ᵀ H d₀ = (−1)(−2) + 0 = 2 ⇒ α₀ = 2/2 = 1.0.<br/>"
        "&nbsp;&nbsp;<b>X₁ = X₀ + α₀ d₀ = [−1, 1]ᵀ</b>.<br/>"
        "<b>Iteration 2:</b> X₁ = (−1, 1)ᵀ ⇒ ∇f(X₁) = [−1, −1]ᵀ ⇒ d₁ = [1, 1]ᵀ.<br/>"
        "&nbsp;&nbsp;d₁ᵀ d₁ = 2; H d₁ = [6, 4]ᵀ; d₁ᵀ H d₁ = 10 ⇒ α₁ = 2/10 = 0.2.<br/>"
        "&nbsp;&nbsp;<b>X₂ = X₁ + α₁ d₁ = [−1, 1]ᵀ + 0.2[1, 1]ᵀ = [−0.8, 1.2]ᵀ</b>.<br/>"
        "<b>Iteration 3:</b> X₂ = (−0.8, 1.2)ᵀ ⇒ ∇f(X₂) = [0.2, −0.2]ᵀ ⇒ d₂ = [−0.2, 0.2]ᵀ.<br/>"
        "&nbsp;&nbsp;d₂ᵀ d₂ = 0.08; H d₂ = [−0.4, 0]ᵀ; d₂ᵀ H d₂ = 0.08 ⇒ α₂ = 0.08/0.08 = 1.0.<br/>"
        "&nbsp;&nbsp;<b>X₃ = X₂ + α₂ d₂ = [−0.8, 1.2]ᵀ + 1.0[−0.2, 0.2]ᵀ = [−1.0, 1.4]ᵀ</b>.",
        body_style
    ))

    # --- Q10 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q10. Multivariable minimization: Nelder–Mead algorithm", q_title_style))
    story.append(Paragraph(
        "f(x, y) = x³ − 4x + y² − xy; V₁ = (0, 0), V₂ = (1.2, 0), V₃ = (0, 0.8). α=1, γ=2, ρ=0.5, σ=0.5 (5 iterations).<br/>"
        "Initial values: f(V₁) = 0 (G), f(V₂) = −3.072 (B), f(V₃) = 0.64 (W).<br/>"
        "• <b>Iteration 1:</b> Centroid of B and G: C = (V₂+V₁)/2 = (0.6, 0).<br/>"
        "&nbsp;&nbsp;Reflection Vᵣ = C + 1(C − W) = (1.2, −0.8) ⇒ f(Vᵣ) = −1.472. Since f(B) &lt; f(Vᵣ) &lt; f(G), <b>Accept Reflection</b>. Simplex: V₁=(0, 0), V₂=(1.2, 0), V₃=(1.2, −0.8).<br/>"
        "• <b>Iteration 2:</b> Best V₂ (−3.072), Good V₃ (−1.472), Worst V₁ (0). Centroid C = (1.2, −0.4).<br/>"
        "&nbsp;&nbsp;Reflection Vᵣ = (2.4, −0.8) ⇒ f(Vᵣ) = 6.784 (&gt; f(W)). Contraction V_c = C − 0.5(C − W) = (0.6, −0.2) ⇒ f(V_c) = −2.024. <b>Accept Contraction</b>. Simplex: V₁=(0.6, −0.2), V₂=(1.2, 0), V₃=(1.2, −0.8).<br/>"
        "• <b>Iteration 3:</b> Best V₂ (−3.072), Good V₁ (−2.024), Worst V₃ (−1.472). Centroid C = (0.9, −0.1).<br/>"
        "&nbsp;&nbsp;Reflection Vᵣ = (0.6, 0.6) ⇒ f(Vᵣ) = −2.184. Since f(B) &lt; f(Vᵣ) &lt; f(G), <b>Accept Reflection</b>. Simplex: V₁=(0.6, −0.2), V₂=(1.2, 0), V₃=(0.6, 0.6).<br/>"
        "• <b>Iteration 4:</b> Best V₂ (−3.072), Good V₃ (−2.184), Worst V₁ (−2.024). Centroid C = (0.9, 0.3).<br/>"
        "&nbsp;&nbsp;Reflection Vᵣ = (1.2, 0.8) ⇒ f(Vᵣ) = −3.392 (&lt; f(B)). Expansion Vₑ = C + 2(Vᵣ − C) = (1.5, 1.3) ⇒ f(Vₑ) = −2.885. Since f(Vᵣ) &lt; f(Vₑ), <b>Accept Reflection Vᵣ=(1.2, 0.8)</b>. Simplex: V₁=(1.2, 0.8), V₂=(1.2, 0), V₃=(0.6, 0.6).<br/>"
        "• <b>Iteration 5:</b> Best V₁ (−3.392), Good V₂ (−3.072), Worst V₃ (−2.184). Centroid C = (1.2, 0.4).<br/>"
        "&nbsp;&nbsp;Reflection Vᵣ = (1.8, 0.2) ⇒ f(Vᵣ) = −1.688 (&gt; f(W)). Contraction V_c = (0.9, 0.5) ⇒ f(V_c) = −3.071. <b>Accept Contraction</b>.<br/>"
        "<b>Final Simplex:</b> V₁ = (1.2, 0.8) [f = −3.392], V₂ = (1.2, 0) [f = −3.072], V₃ = (0.9, 0.5) [f = −3.071].",
        body_style
    ))

    # =========================================================================
    # UNIT II
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("UNIT II | Interpolation and Numerical Integration", unit_header_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#2B6CB0"), spaceBefore=0, spaceAfter=8))

    # --- Q1 ---
    story.append(Paragraph("Q1. Interpolation: assumptions and interpolation-error formula", q_title_style))
    story.append(Paragraph("<b>(a) State assumptions for polynomial interpolation at n+1 distinct nodes; explain uniqueness.</b>", subpart_style))
    story.append(Paragraph(
        "• <b>Assumptions:</b> (i) n+1 distinct points x₀, x₁, ..., xₙ (xᵢ ≠ xⱼ for i ≠ j), (ii) corresponding values yᵢ = f(xᵢ) are known, (iii) interpolating function Pₙ(x) is a polynomial of degree ≤ n satisfying Pₙ(xᵢ) = yᵢ.<br/>"
        "• <b>Uniqueness Proof:</b> Suppose two polynomials Pₙ(x) and Qₙ(x) of degree ≤ n both interpolate the data. Let D(x) = Pₙ(x) − Qₙ(x). D(x) is a polynomial of degree ≤ n. However, D(xᵢ) = yᵢ − yᵢ = 0 at all n+1 distinct nodes. By the Fundamental Theorem of Algebra, a non-zero polynomial of degree ≤ n can have at most n roots. Thus D(x) must be identically zero ⇒ <b>Pₙ(x) ≡ Qₙ(x)</b>.",
        body_style
    ))

    story.append(Paragraph("<b>(b) State remainder Rₙ(x); for f(x) = 1/(1+x) on [0, 1] at x=0, 0.5, 1, bound error at x=0.25.</b>", subpart_style))
    story.append(Paragraph(
        "• <b>Remainder Formula:</b> Rₙ(x) = [f⁽ⁿ⁺¹⁾(ξ) / (n+1)!] · ∏ᵢ₌₀ⁿ (x − xᵢ), where ξ ∈ (min(xᵢ, x), max(xᵢ, x)), requiring f ∈ Cⁿ⁺¹[a, b].<br/>"
        "• For f(x) = (1+x)⁻¹ and n = 2 nodes (x₀=0, x₁=0.5, x₂=1):<br/>"
        "&nbsp;&nbsp;f'''(x) = −6(1+x)⁻⁴ ⇒ R₂(x) = [−6(1+ξ)⁻⁴ / 3!] · x(x − 0.5)(x − 1) = <b>−[x(x − 0.5)(x − 1)] / (1 + ξ)⁴</b>, ξ ∈ (0, 1).<br/>"
        "• At x = 0.25: ω(0.25) = (0.25)(−0.25)(−0.75) = 3/64 = 0.046875.<br/>"
        "Since 0 &lt; ξ &lt; 1, 1/(1+ξ)⁴ ≤ 1. Hence, <b>|R₂(0.25)| ≤ 3/64 ≈ 0.046875</b>.",
        body_style
    ))

    # --- Q2 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q2. Finite differences and their identities", q_title_style))
    story.append(Paragraph("<b>(a) Evaluate Δ³[(1 − x)(1 − 2x)(1 − 3x)] when interval of differencing is unity (h = 1).</b>", subpart_style))
    story.append(Paragraph(
        "Expanding the polynomial: (1 − x)(1 − 2x)(1 − 3x) = −6x³ + 11x² − 6x + 1.<br/>"
        "This is a 3rd degree polynomial with leading coefficient a₃ = −6. By the finite difference property, Δⁿ(xⁿ) = n!·hⁿ, and differences of lower-degree terms vanish.<br/>"
        "<b>Δ³[P(x)] = a₃ · 3! · h³ = (−6) · 6 · (1)³ = −36</b>.",
        body_style
    ))

    story.append(Paragraph("<b>(b) Prove that Δ²y₃ = ∇²y₅. Define forward, backward, central differences and state relations.</b>", subpart_style))
    story.append(Paragraph(
        "• <b>Definitions:</b> Forward Δyₖ = yₖ₊₁ − yₖ ⇒ <b>Δ = E − 1</b>. Backward ∇yₖ = yₖ − yₖ₋₁ ⇒ <b>∇ = 1 − E⁻¹</b>. Central δyₖ = yₖ₊½ − yₖ₋½ = (E½ − E⁻½)yₖ.<br/>"
        "• <b>Proof:</b><br/>"
        "&nbsp;&nbsp;LHS: Δ²y₃ = (E − 1)²y₃ = (E² − 2E + 1)y₃ = y₅ − 2y₄ + y₃.<br/>"
        "&nbsp;&nbsp;RHS: ∇²y₅ = (1 − E⁻¹)²y₅ = (1 − 2E⁻¹ + E⁻²)y₅ = y₅ − 2y₄ + y₃.<br/>"
        "Since LHS = RHS = y₅ − 2y₄ + y₃, <b>Δ²y₃ = ∇²y₅ is proved</b>.",
        body_style
    ))

    # --- Q3 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q3. Gregory–Newton forward interpolation", q_title_style))
    story.append(Paragraph("Estimate f(−1.5) from table (x: −2, −1, 0, 1, 2, 3; y: 15, 5, 1, 3, 11, 25).", body_style))
    
    gn_data = [
        [Paragraph("<b>x</b>", table_cell_bold), Paragraph("<b>y</b>", table_cell_bold), Paragraph("<b>Δy</b>", table_cell_bold), Paragraph("<b>Δ²y</b>", table_cell_bold), Paragraph("<b>Δ³y</b>", table_cell_bold), Paragraph("<b>Δ⁴y</b>", table_cell_bold)],
        ["−2", "15", "−10", "6", "0", "0"],
        ["−1", "5", "−4", "6", "0", ""],
        ["0", "1", "2", "6", "", ""],
        ["1", "3", "8", "", "", ""],
        ["2", "11", "14", "", "", ""],
        ["3", "25", "", "", "", ""]
    ]
    t_gn = Table(gn_data, colWidths=[50, 60, 60, 60, 60, 60])
    t_gn.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_gn)
    story.append(Paragraph(
        "x₀ = −2, h = 1, x = −1.5 ⇒ u = [−1.5 − (−2)] / 1 = 0.5.<br/>"
        "f(−1.5) = y₀ + u·Δy₀ + [u(u−1)/2!]·Δ²y₀ = 15 + (0.5)(−10) + [(0.5)(−0.5)/2]·(6) = 15 − 5 − 0.75 = <b>9.25</b>.",
        body_style
    ))

    # --- Q4 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q4. Gregory–Newton backward interpolation", q_title_style))
    story.append(Paragraph("Estimate f(1.5) from table (x: −1, 0, 1, 2; y: 5, 1, 1, 11).", body_style))
    gnb_data = [
        [Paragraph("<b>x</b>", table_cell_bold), Paragraph("<b>y</b>", table_cell_bold), Paragraph("<b>∇y</b>", table_cell_bold), Paragraph("<b>∇²y</b>", table_cell_bold), Paragraph("<b>∇³y</b>", table_cell_bold)],
        ["−1", "5", "", "", ""],
        ["0", "1", "−4", "", ""],
        ["1", "1", "0", "4", ""],
        ["2", "11", "10", "10", "6"]
    ]
    t_gnb = Table(gnb_data, colWidths=[60, 70, 70, 70, 70])
    t_gnb.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_gnb)
    story.append(Paragraph(
        "xₙ = 2, h = 1, x = 1.5 ⇒ v = (1.5 − 2)/1 = −0.5.<br/>"
        "f(1.5) = yₙ + v·∇yₙ + [v(v+1)/2!]·∇²yₙ + [v(v+1)(v+2)/3!]·∇³yₙ<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= 11 + (−0.5)(10) + [(−0.5)(0.5)/2](10) + [(−0.5)(0.5)(1.5)/6](6) = 11 − 5 − 1.25 − 0.375 = <b>4.375</b>.<br/>"
        "• <b>Convenience:</b> Near the end of a tabulated dataset, forward differences cannot be formed because they require points outside the table boundary. The backward form utilizes readily available preceding points, avoiding out-of-bound errors.",
        body_style
    ))

    # --- Q5 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q5. Lagrange interpolation", q_title_style))
    story.append(Paragraph(
        "Evaluate f(10) from (5, 12), (6, 13), (9, 14), (11, 16).<br/>"
        "• <b>Lagrange Form:</b> P(x) = L₀(x)·12 + L₁(x)·13 + L₂(x)·14 + L₃(x)·16, where:<br/>"
        "&nbsp;&nbsp;L₀(x) = −[(x−6)(x−9)(x−11)] / 24;<br/>"
        "&nbsp;&nbsp;L₁(x) = +[(x−5)(x−9)(x−11)] / 15;<br/>"
        "&nbsp;&nbsp;L₂(x) = −[(x−5)(x−6)(x−11)] / 24;<br/>"
        "&nbsp;&nbsp;L₃(x) = +[(x−5)(x−6)(x−9)] / 60.<br/>"
        "• At x = 10: L₀(10) = 1/6, L₁(10) = −1/3, L₂(10) = 5/6, L₃(10) = 1/3.<br/>"
        "• <b>f(10) = (1/6)(12) − (1/3)(13) + (5/6)(14) + (1/3)(16) = 2 − 13/3 + 35/3 + 16/3 = 44/3 ≈ 14.6667</b>.",
        body_style
    ))

    # --- Q6 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q6. Newton’s divided-difference interpolation", q_title_style))
    story.append(Paragraph("<b>(a) Divided-difference table, interpolating polynomial, and f(0.5):</b>", subpart_style))
    dd_data = [
        [Paragraph("<b>x</b>", table_cell_bold), Paragraph("<b>f(x)</b>", table_cell_bold), Paragraph("<b>1st Div. Diff.</b>", table_cell_bold), Paragraph("<b>2nd Div. Diff.</b>", table_cell_bold), Paragraph("<b>3rd Div. Diff.</b>", table_cell_bold)],
        ["1", "3", "0.5", "1/3 ≈ 0.3333", "−2"],
        ["1.5", "13/4 = 3.25", "1/6 ≈ 0.1667", "−5/3 ≈ −1.6667", ""],
        ["0", "3", "−2/3 ≈ −0.6667", "", ""],
        ["2", "5/3 ≈ 1.6667", "", "", ""]
    ]
    t_dd = Table(dd_data, colWidths=[40, 75, 80, 85, 75])
    t_dd.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_dd)
    story.append(Paragraph(
        "• <b>Polynomial:</b> P(x) = 3 + 0.5(x − 1) + (1/3)(x − 1)(x − 1.5) − 2(x − 1)(x − 1.5)(x − 0).<br/>"
        "• At x = 0.5: P(0.5) = 3 + 0.5(−0.5) + (1/3)(−0.5)(−1) − 2(−0.5)(−1)(0.5) = 3 − 0.25 + 1/6 − 0.5 = <b>29/12 ≈ 2.4167</b>.<br/>"
        "<b>(b) For f(x) = 1/x², find [a, b] and [a, b, c]:</b><br/>"
        "• f[a, b] = [1/b² − 1/a²] / (b − a) = [(a² − b²) / (a²b²)] / (b − a) = <b>−(a + b) / (a²b²)</b>.<br/>"
        "• f[a, b, c] = [f[b, c] − f[a, b]] / (c − a) = [−(b+c)/(b²c²) + (a+b)/(a²b²)] / (c − a)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= [c²(a+b) − a²(b+c)] / [a²b²c²(c − a)] = [(c − a)(ab + bc + ca)] / [a²b²c²(c − a)] = <b>(ab + bc + ca) / (a²b²c²)</b>.",
        body_style
    ))

    # --- Q7 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q7. Definite integrals, Newton–Cotes formulas and trapezoidal rule", q_title_style))
    story.append(Paragraph(
        "<b>(a) Newton–Cotes & Composite Trapezoidal Derivation:</b><br/>"
        "Newton–Cotes formulas replace f(x) by an n-th degree Lagrange interpolating polynomial over equidistant points. For n = 1 on [x₀, x₁], integrating linear P₁(x) gives h/2·(y₀ + y₁). Subdividing [a, b] into n subintervals and summing:<br/>"
        "&nbsp;&nbsp;<b>∫ₐᵇ f(x) dx ≈ (h/2) [y₀ + 2∑ᵢ₌₁ⁿ⁻¹ yᵢ + yₙ]</b>, where h = (b − a)/n.<br/>"
        "<b>(b) Approximate ∫₀³ (x² + 1) dx with n = 6:</b><br/>"
        "h = (3 − 0)/6 = 0.5. Nodes: x₀=0 (1), x₁=0.5 (1.25), x₂=1 (2), x₃=1.5 (3.25), x₄=2 (5), x₅=2.5 (7.25), x₆=3 (10).<br/>"
        "I_trap = (0.5/2) [1 + 10 + 2(1.25 + 2 + 3.25 + 5 + 7.25)] = 0.25 [11 + 2(18.75)] = 0.25(48.5) = <b>12.125</b>.<br/>"
        "Exact integral = [x³/3 + x]₀³ = 9 + 3 = <b>12.0</b>. Absolute error = |12.0 − 12.125| = <b>0.125</b>.",
        body_style
    ))

    # --- Q8 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q8. Simpson’s one-third and three-eighth rules", q_title_style))
    story.append(Paragraph(
        "Evaluate I = ∫₀¹ [1/(1 + x²)] dx.<br/>"
        "• <b>(i) Simpson’s 1/3 rule (h = 1/4 ⇒ n = 4):</b> Nodes: 0, 0.25, 0.5, 0.75, 1.0.<br/>"
        "&nbsp;&nbsp;y₀ = 1, y₁ = 16/17 ≈ 0.941176, y₂ = 0.8, y₃ = 16/25 = 0.64, y₄ = 0.5.<br/>"
        "&nbsp;&nbsp;I_{1/3} = (h/3) [y₀ + y₄ + 4(y₁ + y₃) + 2y₂] = (1/12) [1.5 + 4(1.581176) + 1.6] = <b>0.785392</b>.<br/>"
        "• <b>(ii) Simpson’s 3/8 rule (h = 1/6 ⇒ n = 6):</b> Nodes: 0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.<br/>"
        "&nbsp;&nbsp;y₀=1, y₁=36/37≈0.972973, y₂=0.9, y₃=0.8, y₄=36/52≈0.692308, y₅=36/61≈0.590164, y₆=0.5.<br/>"
        "&nbsp;&nbsp;I_{3/8} = (3h/8) [y₀ + y₆ + 3(y₁ + y₂ + y₄ + y₅) + 2y₃] = (1/16) [1.5 + 3(3.155445) + 1.6] = <b>0.785396</b>.<br/>"
        "• <b>Comparison:</b> Exact value = π/4 ≈ 0.78539816. Error (1/3 rule) ≈ 6 × 10⁻⁶; Error (3/8 rule) ≈ 2 × 10⁻⁶.<br/>"
        "• <b>Subinterval Restrictions:</b> Simpson's 1/3 requires <b>n to be an even number</b> (multiple of 2); Simpson's 3/8 requires <b>n to be a multiple of 3</b>.",
        body_style
    ))

    # --- Q9 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q9. Errors in quadrature formulas", q_title_style))
    story.append(Paragraph(
        "<b>(a) Error Bounds and Requirements:</b><br/>"
        "• Trapezoidal: |E_T| ≤ [(b − a)³ / (12n²)] · M₂, requiring f ∈ C²[a, b] where M₂ = max |f''(x)|.<br/>"
        "• Simpson's 1/3: |E_{S1/3}| ≤ [(b − a)⁵ / (180n⁴)] · M₄, requiring f ∈ C⁴[a, b] where M₄ = max |f⁽⁴⁾(x)|.<br/>"
        "• Simpson's 3/8: |E_{S3/8}| ≤ [(b − a)⁵ / (80n⁴)] · M₄, requiring f ∈ C⁴[a, b] where M₄ = max |f⁽⁴⁾(x)|.<br/>"
        "<b>(b) Subintervals for Error &lt; 10⁻³ in ∫₀¹ [1/(1 + x)] dx via Trapezoidal Rule:</b><br/>"
        "b − a = 1. f''(x) = 2(1 + x)⁻³ ⇒ M₂ = max_{x∈[0, 1]} |f''(x)| = 2/(1 + 0)³ = 2.<br/>"
        "Error bound: [1³ × 2] / [12n²] &lt; 10⁻³ ⇒ 1 / (6n²) &lt; 10⁻³ ⇒ 6n² &gt; 1000 ⇒ n² &gt; 166.67 ⇒ n &gt; 12.91.<br/>"
        "Taking the next integer: <b>n ≥ 13 subintervals</b>.",
        body_style
    ))

    # --- Q10 ---
    story.append(Spacer(1, 6))
    story.append(Paragraph("Q10. Romberg integration and Gaussian quadrature", q_title_style))
    story.append(Paragraph(
        "<b>(a) Romberg Integration for ∫₀¹ [1/(1 + x)] dx to 3 decimal places:</b><br/>"
        "Trapezoidal estimates: R₁,₁ = 0.5[1 + 0.5] = 0.7500; R₂,₁ = 0.5(0.75) + 0.5(1/1.5) = 0.7083; R₃,₁ = 0.5(0.7083) + 0.25[0.8 + 0.5714] = 0.6970.<br/>"
        "Richardson extrapolation: R_{k, j} = [4ʲ⁻¹ R_{k, j-1} − R_{k-1, j-1}] / [4ʲ⁻¹ − 1].",
        body_style
    ))
    rom_data = [
        [Paragraph("<b>k</b>", table_cell_bold), Paragraph("<b>h</b>", table_cell_bold), Paragraph("<b>R_{k, 1} (Trapezoidal)</b>", table_cell_bold), Paragraph("<b>R_{k, 2} (Simpson)</b>", table_cell_bold), Paragraph("<b>R_{k, 3} (Boole)</b>", table_cell_bold)],
        ["1", "1", "0.7500", "", ""],
        ["2", "1/2", "0.7083", "0.6944", ""],
        ["3", "1/4", "0.6970", "0.6933", "0.6932"]
    ]
    t_rom = Table(rom_data, colWidths=[30, 45, 115, 115, 115])
    t_rom.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_rom)
    story.append(Paragraph(
        "To 3 decimal places: <b>I ≈ 0.693</b> (Exact ln 2 ≈ 0.693147).<br/>"
        "<b>(b) Three-point Gauss–Legendre for ∫₋₂² e^(−x²) dx:</b><br/>"
        "• Transformation to [−1, 1]: x = 2t ⇒ dx = 2 dt ⇒ I = 2 ∫₋₁¹ e^(−4t²) dt.<br/>"
        "• Nodes & Weights on [−1, 1]: t₁ = −√(3/5) ≈ −0.774597 (w₁ = 5/9), t₂ = 0 (w₂ = 8/9), t₃ = +√(3/5) (w₃ = 5/9).<br/>"
        "• Function g(t) = 2e^(−4t²):<br/>"
        "&nbsp;&nbsp;g(t₁) = 2e^(−2.4) ≈ 0.181436; g(t₂) = 2(1) = 2.0; g(t₃) = 2e^(−2.4) ≈ 0.181436.<br/>"
        "• Quadrature sum: I ≈ (5/9)(0.181436) + (8/9)(2.0) + (5/9)(0.181436) = (10/9)(0.181436) + 16/9 = <b>1.9794</b>.",
        body_style
    ))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {PDF_FILENAME}")

if __name__ == "__main__":
    build_pdf()
