"""Generate the official 2-page brief PDF for Hackathon submission.

Page 1: Project overview, system architecture, and 6-block model design.
Page 2: Training details, API contract compliance, and results.
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
)

ROOT_DIR = Path(__file__).resolve().parent.parent


def create_brief_pdf(filename: str = "Multiverse_of_Monsoon_Brief.pdf") -> str:
    pdf_path = (ROOT_DIR / filename).resolve()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=28,
        rightMargin=28,
        topMargin=22,
        bottomMargin=22,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569"),
        spaceAfter=5,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=5,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.8,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=2,
    )
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
        spaceAfter=1.5,
    )
    cell_style = ParagraphStyle(
        "CellText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.0,
        leading=9.0,
        textColor=colors.HexColor("#1e293b"),
    )
    cell_bold = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.0,
        leading=9.0,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("Multiverse of Monsoon — Climate Digital Twin", title_style))
    story.append(
        Paragraph(
            "<b>Track 2: Earth-System Modelling &amp; Climate Simulation</b> | Team Vortex | "
            "System Architecture &amp; Design Brief",
            subtitle_style,
        )
    )

    # 1. Project Overview
    story.append(Paragraph("1. Project Overview", h1_style))
    story.append(
        Paragraph(
            "Multiverse of Monsoon is a physics-aware deep learning engine that simulates Indian monsoon dynamics "
            "over a <b>32x36 grid at 1 degree resolution</b> (latitudes 6.5-37.5N, longitudes 65.5-100.5E) "
            "across six atmospheric channels: temperature (t2m), precipitation (tp), zonal wind (u10), "
            "meridional wind (v10), sea-level pressure (msl), and total column water vapour (tcwv). "
            "Given a 3-day observational history, the engine autoregressively rolls weather forward up to 10 days. "
            "It also supports counterfactual branching — injecting a localized physical perturbation to model "
            "how an anomaly propagates across the subcontinent.",
            body_style,
        )
    )

    story.append(Spacer(1, 2))

    # 2. System Architecture
    story.append(Paragraph("2. System Architecture", h1_style))
    story.append(
        Paragraph(
            "The submission is structured as three decoupled, offline-only components communicating over localhost:",
            body_style,
        )
    )

    arch_data = [
        [
            Paragraph("<b>Component</b>", cell_bold),
            Paragraph("<b>File(s)</b>", cell_bold),
            Paragraph("<b>Role</b>", cell_bold),
        ],
        [
            Paragraph("Inference Server", cell_style),
            Paragraph("serve.py (port 8000)", cell_style),
            Paragraph(
                "FastAPI server. Loads weights.pt and static.pt at startup. Exposes GET/POST /health "
                "and POST /simulate. Accepts init_states [3,6,32,36], doy_sincos [2], steps, and an "
                "optional perturbation [6,32,36]. Returns rollout [steps,6,32,36] and inference_ms. "
                "Fully offline — no external network calls.",
                cell_style,
            ),
        ],
        [
            Paragraph("Training Script", cell_style),
            Paragraph("train.py", cell_style),
            Paragraph(
                "Entry point. Reads hackdata/india_climate_train.nc via HDF5. Builds 3-day sliding "
                "windows [B,3,6,32,36] with next-day targets [B,6,32,36]. Trains the 6-block model "
                "with AdamW and channel-normalised MSE loss. Saves weights.pt and static.pt.",
                cell_style,
            ),
        ],
        [
            Paragraph("Interactive Dashboard", cell_style),
            Paragraph("dashboard/app.py (port 8501)", cell_style),
            Paragraph(
                "Separate FastAPI server. Calls serve.py /simulate to run a baseline rollout and an "
                "optional perturbed rollout, then computes the difference. Serves a single-page "
                "HTML/JS interface (dashboard/templates/index.html) with canvas-rendered heatmaps "
                "for all 6 channels, a day scrubber (Day 1-10), two preset shocks, and a click-drag "
                "custom shock studio. Also computes regional impact metrics (dashboard/impact.py).",
                cell_style,
            ),
        ],
    ]

    t_arch = Table(arch_data, colWidths=[90, 100, 362])
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
    story.append(Spacer(1, 4))

    # 3. 6-Block Model Design
    story.append(Paragraph("3. The 6-Block Deep Learning Pipeline", h1_style))
    story.append(
        Paragraph(
            "The core model (monsoon_r1/models/multiverse_model.py) is a strictly sequential 6-block architecture. "
            "Each block is a separate PyTorch nn.Module. Data flows in one direction; "
            "there is no skip-connection between non-adjacent blocks.",
            body_style,
        )
    )

    model_data = [
        [
            Paragraph("<b>Block</b>", cell_bold),
            Paragraph("<b>Module</b>", cell_bold),
            Paragraph("<b>Input Shape</b>", cell_bold),
            Paragraph("<b>Output Shape</b>", cell_bold),
            Paragraph("<b>What it does</b>", cell_bold),
        ],
        [
            Paragraph("1", cell_style),
            Paragraph("ClimateInputEncoder", cell_style),
            Paragraph("[B, 3, 6, 32, 36] + [B, 2, 32, 36] + [B, 2]", cell_style),
            Paragraph("[B, 3, 48, 32, 36]", cell_style),
            Paragraph(
                "Three 1x1 convolutions project (a) each day's 6 climate channels to 32 features, "
                "(b) the 2 static geography fields (land-sea mask + orography) to 8 features broadcast "
                "across all 3 days, and (c) the 2 day-of-year sincos scalars to 8 features broadcast "
                "across all grid cells. Concatenated to 48 channels. Inputs are zero-mean unit-variance "
                "normalised using stats.json before projection.",
                cell_style,
            ),
        ],
        [
            Paragraph("2", cell_style),
            Paragraph("MultiScaleSpatialEncoder", cell_style),
            Paragraph("[B, 3, 48, 32, 36]", cell_style),
            Paragraph("[B, 3, 64, 32, 36]", cell_style),
            Paragraph(
                "Three parallel conv branches per day: 3x3 (local), 5x5 (synoptic), and 3x3 dilated "
                "dilation=2 (wide context), each producing 16 channels. Concatenated (48 ch) then "
                "fused to 64 channels via 1x1 conv.",
                cell_style,
            ),
        ],
        [
            Paragraph("3", cell_style),
            Paragraph("TemporalDynamicsEncoder (ConvLSTM)", cell_style),
            Paragraph("[B, 3, 64, 32, 36]", cell_style),
            Paragraph("[B, 64, 32, 36]", cell_style),
            Paragraph(
                "A single ConvLSTMCell steps through the 3-day sequence. Standard LSTM gates "
                "(input, forget, cell, output) are computed by 2D convolutions over both the "
                "current spatial features and the previous hidden state. Returns the final hidden "
                "state H_T representing compressed temporal evolution.",
                cell_style,
            ),
        ],
        [
            Paragraph("4", cell_style),
            Paragraph("SpatialContextAttention (CBAM-style)", cell_style),
            Paragraph("[B, 64, 32, 36]", cell_style),
            Paragraph("[B, 64, 32, 36]", cell_style),
            Paragraph(
                "Channel attention: shared MLP over global avg+max pool, sigmoid gate reweights "
                "channels. Spatial attention: channel-wise avg+max concatenated, 7x7 conv, sigmoid "
                "gate reweights spatial positions. Applied sequentially; shape is preserved.",
                cell_style,
            ),
        ],
        [
            Paragraph("5", cell_style),
            Paragraph("ForecastDecoder", cell_style),
            Paragraph("[B, 64, 32, 36]", cell_style),
            Paragraph("[B, 6, 32, 36]", cell_style),
            Paragraph(
                "Three conv layers: 64->64 (3x3, ReLU), 64->32 (3x3, ReLU), 32->6 (1x1, linear). "
                "Outputs delta S — the predicted change in each atmospheric channel for the next day.",
                cell_style,
            ),
        ],
        [
            Paragraph("6", cell_style),
            Paragraph("ResidualStateTransition", cell_style),
            Paragraph("[B, 6, 32, 36] (state + delta)", cell_style),
            Paragraph("[B, 6, 32, 36]", cell_style),
            Paragraph(
                "S_next = S_curr + delta_S * sigma, where sigma are the per-channel ERA5 standard "
                "deviations from stats.json. Applies an out-of-place clamp enforcing tp >= 0.0 "
                "(precipitation cannot be negative). The resulting state is the next day's forecast "
                "and slides into the autoregressive window for the following step.",
                cell_style,
            ),
        ],
    ]

    t_model = Table(model_data, colWidths=[28, 105, 100, 70, 249])
    t_model.setStyle(
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
    story.append(t_model)

    # ==================== PAGE 2 ====================
    story.append(PageBreak())

    story.append(Paragraph("Multiverse of Monsoon — Training, API Contract &amp; Evaluation", title_style))
    story.append(
        Paragraph(
            "<b>Track 2: Earth-System Modelling &amp; Climate Simulation</b> | Team Vortex | "
            "Training Design, Compliance &amp; Verification",
            subtitle_style,
        )
    )

    # 1. Dataset and Training
    story.append(Paragraph("1. Dataset &amp; Training", h1_style))
    story.append(
        Paragraph(
            "Training data: <b>hackdata/india_climate_train.nc</b> — 6,209 daily ERA5 observations (2005–2021), "
            "6 channels, 32x36 grid. Split chronologically: days 0–5843 for training, days 5844–end (year 2021) "
            "withheld as the holdout evaluation set. Static geography (land-sea mask and orography) is extracted "
            "from the NetCDF file and saved to static.pt and hackdata/static.npy. Per-channel means and standard "
            "deviations are saved to hackdata/stats.json and loaded by both train.py and serve.py.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Input window:</b> 3 consecutive days [B, 3, 6, 32, 36] + static geography [B, 2, 32, 36] "
            "+ day-of-year sincos [B, 2].",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Target:</b> Next day's physical state [B, 6, 32, 36]. The model predicts a delta; "
            "the loss is MSE on each channel normalised by that channel's ERA5 variance (sigma_c^2) so "
            "all 6 channels contribute equally regardless of physical magnitude.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Optimiser:</b> AdamW, lr = 1e-3, trained for 3 epochs, batch size 32.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Output files:</b> weights.pt (trained model checkpoint, 1.57 MB) and static.pt "
            "[2, 32, 36] geographic tensor, both stored in the repository root.",
            bullet_style,
        )
    )

    story.append(Spacer(1, 3))

    # 2. API Contract Compliance
    story.append(Paragraph("2. API Contract Compliance", h1_style))
    story.append(
        Paragraph(
            "serve.py implements the official Round 1 API contract as defined in monsoon_r1/contract.py "
            "and verified by monsoon_r1/schema_validator.py:",
            body_style,
        )
    )

    api_data = [
        [
            Paragraph("<b>Requirement</b>", cell_bold),
            Paragraph("<b>Implementation</b>", cell_bold),
            Paragraph("<b>Status</b>", cell_bold),
        ],
        [
            Paragraph("serve.py at repo root, binds 0.0.0.0:8000", cell_style),
            Paragraph("serve.py starts with uvicorn on 0.0.0.0, default port 8000", cell_style),
            Paragraph("Met", cell_style),
        ],
        [
            Paragraph("GET /health returns {status: ready, model_name: ...}", cell_style),
            Paragraph("Returns JSON with status and model_name=multiverse-of-monsoon-r1", cell_style),
            Paragraph("Met", cell_style),
        ],
        [
            Paragraph("POST /simulate accepts init_states, doy_sincos, steps, perturbation", cell_style),
            Paragraph(
                "Accepts nested JSON lists (canonical) or base64 float32 dict. "
                "Validates shape [3,6,32,36] for init_states, [6,32,36] for perturbation.",
                cell_style,
            ),
            Paragraph("Met", cell_style),
        ],
        [
            Paragraph("Returns rollout [steps,6,32,36] and inference_ms", cell_style),
            Paragraph("Returns encode_list(rollout) and wall-clock ms", cell_style),
            Paragraph("Met", cell_style),
        ],
        [
            Paragraph("10-step rollout within 5.0 s on 4 CPU / 6 GB RAM", cell_style),
            Paragraph("~140 ms on CPU (measured via scripts/eval.py and scripts/verify_server.py)", cell_style),
            Paragraph("Met", cell_style),
        ],
        [
            Paragraph("Offline — no external internet calls at inference", cell_style),
            Paragraph("All weights, stats, and geography are local. No requests to external hosts.", cell_style),
            Paragraph("Met", cell_style),
        ],
        [
            Paragraph("train.py at repo root", cell_style),
            Paragraph("train.py delegates to monsoon_r1/train.py. Trains on hackdata/ only.", cell_style),
            Paragraph("Met", cell_style),
        ],
        [
            Paragraph("Physical validity: tp >= 0, t2m in [-40, +60] C, no NaN/Inf", cell_style),
            Paragraph("Block 6 clamps tp >= 0. scoring.py validity() checks bounds on every rollout.", cell_style),
            Paragraph("Met", cell_style),
        ],
        [
            Paragraph("Null perturbation must not change output (|delta|/sigma <= 1e-3)", cell_style),
            Paragraph("Zero perturbation leaves init_states unchanged; model is deterministic.", cell_style),
            Paragraph("Met", cell_style),
        ],
        [
            Paragraph("Causal perturbation: energy localised near shock on Day 1 (>= 60%)", cell_style),
            Paragraph(
                "warm_ocean (+3K Bay of Bengal) and monsoon_stall (-40% Arabian Sea winds) "
                "are defined in monsoon_r1/scoring.py make_shock(); causality verified in eval.py.",
                cell_style,
            ),
            Paragraph("Met", cell_style),
        ],
    ]

    t_api = Table(api_data, colWidths=[165, 300, 87])
    t_api.setStyle(
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
    story.append(t_api)
    story.append(Spacer(1, 4))

    # 3. Evaluation Results
    story.append(Paragraph("3. Evaluation Results (Year 2021 Holdout)", h1_style))
    story.append(
        Paragraph(
            "Evaluation is run via scripts/eval.py against the Year 2021 chronological holdout "
            "(days 5844+) using latitude-weighted RMSE skill vs persistence and climatology, "
            "spatial variance ratio, and perturbation causality checks as defined in monsoon_r1/scoring.py:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Day-4 skill vs persistence:</b> mean +0.0787 across all 6 channels "
            "(Day 1: +0.089, Day 4: +0.098, Day 8: +0.085; tp channel: +0.233). "
            "Passes the Day-4 gate (spec: &gt; 0).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Anti-blur (Day 8 spatial variance ratio):</b> 0.88–1.09 across all channels "
            "(spec: 0.40–2.50). Fields retain spatial structure and do not collapse to flat averages.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Perturbation causality:</b> 100% of Day-1 response energy localised near the shock "
            "for both warm_ocean and monsoon_stall (spec: &gt;= 60%).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Null shock consistency:</b> max |delta|/sigma = 0.00e+00 (spec: &lt;= 1e-3). "
            "Zero perturbation produces identical output.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Test suite:</b> 110/110 unit and integration tests pass (pytest -v).",
            bullet_style,
        )
    )

    story.append(Spacer(1, 3))

    # 4. Known Limitations
    story.append(Paragraph("4. Known Limitations", h1_style))
    story.append(
        Paragraph(
            "• <b>Closed domain:</b> The model only covers 6.5–37.5N, 65.5–100.5E. Weather systems "
            "entering from outside this boundary (e.g. equatorial westerlies from south of 6.5N) "
            "cannot be refreshed and degrade slowly over long horizons.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>1-degree resolution:</b> Sub-grid convective events and orographic extremes "
            "(&gt;50 mm/day) are averaged across 1x1-degree cells and may appear smoother "
            "than observed radar composites.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Pressure skill at Day 8:</b> msl skill vs persistence is -0.008 at Day 8 "
            "due to unmodeled boundary pressure gradients.",
            bullet_style,
        )
    )

    doc.build(story)
    print(f"Successfully generated brief PDF at: {pdf_path}")
    return str(pdf_path)


if __name__ == "__main__":
    create_brief_pdf()
