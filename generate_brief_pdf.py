"""Generate the official 2-page brief PDF for Hackathon submission.

Page 1: Model design, drift & blur prevention, and failure modes.
Page 2: Visualization dashboard, impact extension, and policy co-pilot.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)


def create_brief_pdf(filename: str = "Multiverse_of_Monsoon_Brief.pdf") -> str:
    pdf_path = Path(filename).resolve()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=8,
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0369a1"),
        spaceBefore=4,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3,
    )

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("Multiverse of Monsoon — Climate Digital Twin", title_style))
    story.append(
        Paragraph(
            "<b>Track 2: Earth-System Modelling & Climate Simulation</b> | Team Vortex | Model Design & Architectural Brief",
            subtitle_style,
        )
    )

    story.append(Paragraph("1. Architecture Overview & Execution Flow", h1_style))
    story.append(
        Paragraph(
            "The Multiverse of Monsoon AI engine simulates the non-linear spatiotemporal dynamics of the Indian climate system "
            "over a 32×36 grid (1° resolution, centres 6.5°N–37.5°N, 65.5°E–100.5°E) across six core atmospheric channels (t2m, tp, u10, v10, msl, tcwv). "
            "The architecture consists of six strictly sequential blocks executing autoregressive rollouts and counterfactual shock simulations:",
            body_style,
        )
    )

    # Architectural table
    arch_data = [
        [
            Paragraph("<b>Block</b>", body_style),
            Paragraph("<b>Component Name</b>", body_style),
            Paragraph("<b>Tensor Dimensions</b>", body_style),
            Paragraph("<b>Physical Mechanism & Role</b>", body_style),
        ],
        [
            Paragraph("<b>Block 1</b>", body_style),
            Paragraph("Climate Input Encoder", body_style),
            Paragraph("[B, 3, 48, 32, 36]", body_style),
            Paragraph("Multi-modal fusion: 6 climate channels (32) + 2 static geography fields (8) + 2 calendar harmonics (8).", body_style),
        ],
        [
            Paragraph("<b>Block 2</b>", body_style),
            Paragraph("Multi-Scale Spatial Encoder", body_style),
            Paragraph("[B, 3, 64, 32, 36]", body_style),
            Paragraph("Parallel 3×3, 5×5, and dilated 3×3 (d=2) convolutions capturing localized convection to synoptic depressions.", body_style),
        ],
        [
            Paragraph("<b>Block 3</b>", body_style),
            Paragraph("Temporal Dynamics (ConvLSTM)", body_style),
            Paragraph("[B, 64, 32, 36]", body_style),
            Paragraph("Recurrent spatiotemporal cell modeling baroclinic momentum and moisture flux memory across 3-day history.", body_style),
        ],
        [
            Paragraph("<b>Block 4</b>", body_style),
            Paragraph("Spatial Context Attention (CBAM)", body_style),
            Paragraph("[B, 64, 32, 36]", body_style),
            Paragraph("Channel MLP & 7×7 spatial attention dynamically re-weighting active monsoonal troughs and Western Ghats topography.", body_style),
        ],
        [
            Paragraph("<b>Block 5</b>", body_style),
            Paragraph("Forecast Decoder", body_style),
            Paragraph("[B, 6, 32, 36]", body_style),
            Paragraph("Hierarchical convolutional projection decoding latent features into raw physical atmospheric deltas (ΔS).", body_style),
        ],
        [
            Paragraph("<b>Block 6</b>", body_style),
            Paragraph("Residual State Transition", body_style),
            Paragraph("[B, 6, 32, 36]", body_style),
            Paragraph("Residual state update (S_next = S_curr + ΔS) enforcing strict non-negativity constraint on precipitation.", body_style),
        ],
    ]
    t_arch = Table(arch_data, colWidths=[55, 120, 105, 260])
    t_arch.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t_arch)
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. Preventing Drift, Explosion, and Blurry Collapse", h1_style))
    story.append(
        Paragraph(
            "Autoregressive climate models commonly fail through two failure modes: <b>spatial blur</b> (converging to flat climatology) "
            "or <b>numerical runaway explosion</b>. Multiverse of Monsoon resolves these via four rigorous design choices:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Residual Delta Learning (ΔS Formulation):</b> Instead of predicting absolute future fields directly (which encourages the network "
            "to minimize mean squared error by outputting washed-out averages), the network models high-frequency differential change: <i>S<sub>t+1</sub> = S<sub>t</sub> + ΔS</i>. "
            "This preserves localized gradients, sharp coastal fronts, and high-intensity rainfall cores.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Channel-Standard-Deviation Loss Weighting:</b> Optimization uses variance-normalized loss <i>L = (1/C) Σ (MSE<sub>c</sub> / σ<sub>c</sub><sup>2</sup>)</i> "
            "based on empirical ERA5 statistics. This prevents high-magnitude pressure fields (100,000 Pa) from dominating delicate precipitation (0–50 mm/day) or wind (0–15 m/s) dynamics.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Physics-Constrained State Projection:</b> Block 6 applies an exact out-of-place clamp enforcing non-negative precipitation (<i>tp ≥ 0.0 mm/day</i>) "
            "and strictly verifies thermal bounds (<i>t2m ∈ [-40°C, +60°C]</i>). This prevents negative rain artifacts and numerical divergence over long multi-step rollouts.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Static Orographic Anchoring:</b> Surface elevation and land-sea boundaries are injected at every temporal step in Block 1, "
            "ensuring that topographically driven orographic rainfall along the Western Ghats and Himalayan foothills does not drift laterally.",
            bullet_style,
        )
    )

    story.append(Spacer(1, 4))
    story.append(Paragraph("3. Known Failure Modes & Operational Boundaries", h1_style))
    story.append(
        Paragraph(
            "• <b>Extreme Out-of-Distribution Shocks (>15 K anomalies):</b> Shocks exceeding historical ERA5 extremes can stress convolutional receptive fields, "
            "causing damping after day 8.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Extended Horizons (>14 days without global teleconnections):</b> As a regional model over India, unforced rollouts beyond 14 days "
            "naturally lose skill due to unmodeled planetary Rossby waves entering from outside the 65.5°E–100.5°E domain.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Sub-Grid Cloud Microphysics:</b> Highly localized convective cloud bursts (<50 km scale) are represented at the 1° grid resolution scale.",
            bullet_style,
        )
    )

    # ==================== PAGE 2 ====================
    story.append(PageBreak())

    story.append(Paragraph("Multiverse of Monsoon — Visualization & Impact Extension", title_style))
    story.append(
        Paragraph(
            "<b>Interactive Map Dashboard, Multiverse Branching & Climate Policy Co-Pilot</b>",
            subtitle_style,
        )
    )

    story.append(Paragraph("1. Interactive Map Dashboard & Multiverse Shock Simulator", h1_style))
    story.append(
        Paragraph(
            "The system includes a dedicated, separate interactive web dashboard (<b>dashboard/app.py</b>) designed for live desk review "
            "and operational decision-makers. The dashboard is <b>live-coupled</b> to the <code>/simulate</code> engine at <code>http://localhost:8000</code>, "
            "triggering real-time model rollouts on demand in ~140 ms without pre-rendered videos or mock data.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>High-Fidelity 6-Channel Grid Visualization:</b> Canvas-rendered heatmaps of India with dynamic color palettes and min/max/mean metrics "
            "for Temperature (°C), Precipitation (mm/day), Zonal/Meridional Winds (m/s), Pressure (hPa), and Total Column Water Vapor (kg/m²).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Interactive Timeline Scrubber:</b> Day-by-day animation controller allowing users to scrub from Day 1 to Day 10 to inspect "
            "monsoon onset, depression tracks, and rain distribution.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Multiverse Branching & Counterfactual Shock Studio:</b> Users can select or inject localized physical shocks into the climate system: "
            "<br/>&nbsp;&nbsp;1. <i>Bay of Bengal Marine Heatwave:</i> +3.0 K ocean anomaly, showing enhanced cyclonic depression and coastal rain. "
            "<br/>&nbsp;&nbsp;2. <i>Weak Arabian Sea Wind:</i> -5.0 m/s reduction in cross-equatorial Somali jet, modeling monsoon delay. "
            "<br/>&nbsp;&nbsp;3. <i>Pre-Monsoon Land Heat Dome:</i> +4.0 K northern India heatwave. "
            "<br/>&nbsp;&nbsp;4. <i>Difference Map (Branch vs. Prime):</i> Real-time anomaly subtraction displaying causal wave propagation.",
            bullet_style,
        )
    )

    story.append(Spacer(1, 6))
    story.append(Paragraph("2. Socio-Economic Impact Extension: Monsoon Disaster Co-Pilot", h1_style))
    story.append(
        Paragraph(
            "Passing raw tensor predictions to policymakers is insufficient. The <b>Monsoon Disaster Co-Pilot (dashboard/impact.py)</b> "
            "translates 10-day physical trajectories into quantitative socio-economic risk metrics and plain-English actionable advisories:",
            body_style,
        )
    )

    # Impact table
    impact_data = [
        [
            Paragraph("<b>Impact Dimension</b>", body_style),
            Paragraph("<b>Scientific Formulation</b>", body_style),
            Paragraph("<b>Actionable Policy & Societal Value</b>", body_style),
        ],
        [
            Paragraph("<b>Kharif Crop Moisture Stress</b>", body_style),
            Paragraph("Rain deficit & dry spell tracking (days with rain < 1mm) over Indo-Gangetic & Deccan agricultural breadbaskets.", body_style),
            Paragraph("Alerts farmers to delay Kharif sowing or release canal water reserves during extended monsoon breaks, preventing crop failure.", body_style),
        ],
        [
            Paragraph("<b>Extreme Flash Flood Alerts</b>", body_style),
            Paragraph("Cumulative 10-day precipitation thresholding (>150mm & >300mm inundation index) across 32×36 grid cells.", body_style),
            Paragraph("Provides 48–96 hour lead times for State Disaster Management (SDMA) to pre-position NDRF teams and manage dam discharges.", body_style),
        ],
        [
            Paragraph("<b>Coastal Cyclone & Wind Hazard</b>", body_style),
            Paragraph("Cyclonic wind shear (sqrt(u² + v²) > 15 m/s) coupled with localized low pressure depressions in coastal waters.", body_style),
            Paragraph("Automated alerts for maritime fishers and port authorities along the Odisha, Andhra Pradesh, and Gujarat coastlines.", body_style),
        ],
        [
            Paragraph("<b>Clean Wind Energy Potential</b>", body_style),
            Paragraph("Turbine-height kinetic flux estimation via <i>0.5 · ρ · v³</i> across major wind corridors in Tamil Nadu and Gujarat.", body_style),
            Paragraph("Aids renewable energy grid operators in balancing thermal power dispatch with wind power availability.", body_style),
        ],
    ]
    t_impact = Table(impact_data, colWidths=[120, 170, 250])
    t_impact.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t_impact)
    story.append(Spacer(1, 6))

    story.append(Paragraph("3. Automated Policy Co-Pilot Recommendation Engine", h1_style))
    story.append(
        Paragraph(
            "The impact co-pilot automatically synthesizes 10-day trajectory statistics into unambiguous, automated alerts for District Collectors "
            "and NDMA coordinators, e.g.:<br/>"
            "<i>• 'AGRI-ADVISORY: Stalled monsoon conditions detected with extended dry spell (>5 days) in Deccan Plateau. Advise farmers to activate micro-irrigation.'<br/>"
            "• 'FLASH FLOOD ALERT: Severe inundation risk (>150mm accumulated rainfall) detected in Western Ghats. Pre-position disaster rescue teams.'</i>",
            body_style,
        )
    )

    doc.build(story)
    print(f"Successfully generated 2-page brief PDF at: {pdf_path}")
    return str(pdf_path)


if __name__ == "__main__":
    create_brief_pdf()
