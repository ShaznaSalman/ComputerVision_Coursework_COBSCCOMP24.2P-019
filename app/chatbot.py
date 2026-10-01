"""Clinical chatbot: knowledge base plus router, knowledge and governance chat agents."""

from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from app.reports import describe_reported_metrics
from core.config import AppConfig

# Loaded once at startup from the notebook's saved test metrics (no hard-coded results).
_REPORTED_METRICS_TEXT = describe_reported_metrics()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Clinical Chatbot Knowledge Base & Responder
# ─────────────────────────────────────────────────────────────────────────────
_CLINICAL_KB: List[Dict[str, Any]] = [
    # ── DR Stage Descriptions ─────────────────────────────────────────────────
    {
        "keys": ["what are the stages", "stages of diabetic retinopathy", "dr stages", "stages", "grades"],
        "reply": (
            "**Diabetic Retinopathy Stages**\n\n"
            "- **Stage 0 — No DR:** No visible signs of diabetic retinopathy.\n"
            "- **Stage 1 — Mild NPDR:** Small changes such as microaneurysms may be present.\n"
            "- **Stage 2 — Moderate NPDR:** More retinal changes, including bleeding or leakage, may appear.\n"
            "- **Stage 3 — Severe NPDR:** There are more widespread changes and a higher risk of progression.\n"
            "- **Stage 4 — Proliferative DR:** New, fragile blood vessels may grow and require urgent specialist review.\n\n"
            "A qualified eye specialist should confirm the stage and recommend follow-up."
        ),
    },
    {
        "keys": ["stage 0", "no dr", "normal", "healthy"],
        "reply": (
            "**Stage 0 — No Diabetic Retinopathy (No DR)**\n\n"
            "The fundus appears normal with no microvascular abnormalities. "
            "No retinal lesions, hemorrhages, or exudates are detected.\n\n"
            "**Management (AAO PPP (Flaxel et al., 2020)):** Annual dilated fundoscopy screening. "
            "Reinforce glycemic control (HbA1c < 7%), blood pressure < 130/80 mmHg, "
            "and lipid optimization. No treatment intervention required."
        ),
    },
    {
        "keys": ["stage 1", "mild", "mild npdr", "microaneurysm"],
        "reply": (
            "**Stage 1 — Mild Non-Proliferative Diabetic Retinopathy (Mild NPDR)**\n\n"
            "Characterized by the presence of **microaneurysms only** — outpouchings "
            "in fragile retinal capillary walls caused by pericyte degeneration.\n\n"
            "**Management (AAO PPP (Flaxel et al., 2020)):** Annual dilated fundoscopy. Intensify "
            "systemic risk factor control. No intraocular treatment is indicated at this stage."
        ),
    },
    {
        "keys": ["stage 2", "moderate", "moderate npdr", "exudate", "cotton wool"],
        "reply": (
            "**Stage 2 — Moderate Non-Proliferative Diabetic Retinopathy (Moderate NPDR)**\n\n"
            "More than microaneurysms present: dot-and-blot hemorrhages, hard exudates "
            "(lipid leakage), and cotton-wool spots (nerve fiber layer infarcts) visible. "
            "Does not meet the criteria for Severe NPDR.\n\n"
            "**Management (AAO PPP (Flaxel et al., 2020)):** 6–12 month follow-up. Ophthalmologist referral "
            "recommended. Evaluate for clinically significant diabetic macular edema (CSME)."
        ),
    },
    {
        "keys": ["stage 3", "severe", "severe npdr", "4-2-1", "venous beading", "irma"],
        "reply": (
            "**Stage 3 — Severe Non-Proliferative Diabetic Retinopathy (Severe NPDR)**\n\n"
            "Defined by the **4-2-1 rule**: >20 intraretinal hemorrhages in all 4 quadrants, "
            "venous beading in ≥2 quadrants, or prominent intraretinal microvascular "
            "abnormalities (IRMA) in ≥1 quadrant.\n\n"
            "**Management (AAO PPP (Flaxel et al., 2020)):** 3–4 month follow-up with retinal specialist. "
            "High risk of progression to proliferative disease. Consider panretinal "
            "photocoagulation (PRP) prophylactically in high-risk patients."
        ),
    },
    {
        "keys": ["stage 4", "proliferative", "pdr", "neovascularization", "vitreous hemorrhage", "nvd", "nve"],
        "reply": (
            "**Stage 4 — Proliferative Diabetic Retinopathy (PDR)**\n\n"
            "The most advanced stage. VEGF-driven **neovascularization** breaches the "
            "internal limiting membrane producing fragile new vessels on the disc (NVD) "
            "or retina (NVE). Untreated, this leads to vitreous hemorrhage, fibrovascular "
            "proliferation, tractional retinal detachment, and irreversible blindness.\n\n"
            "**Management (AAO PPP (Flaxel et al., 2020)):** Urgent ophthalmologist referral within 1–2 weeks. "
            "Panretinal photocoagulation (PRP) or intravitreal anti-VEGF injections "
            "(ranibizumab, bevacizumab) are first-line treatments."
        ),
    },
    # ── Model Architecture FAQs ───────────────────────────────────────────────
    {
        "keys": ["efficientnet", "backbone", "architecture", "model", "network", "cnn"],
        "reply": (
            "**EfficientNetB3 — Deep Learning Backbone**\n\n"
            "RetinaTrace uses **EfficientNetB3** pre-trained on ImageNet as its convolutional "
            "backbone. EfficientNet applies **compound scaling** — scaling depth, width and "
            "resolution together — which gives strong ImageNet accuracy for its size. The project's "
            "notebook also trains ResNet-50, MobileNetV2, EfficientNetB0, DenseNet-121, VGG-16 and a "
            "CNN from scratch on the same data as comparison trials.\n\n"
            "A custom classification head is added:\n"
            "`GAP → BatchNorm → Dense(256, ReLU) → Dropout(0.5) → Dense(5, Softmax)`\n\n"
            "The notebook trains it in **2 phases** on APTOS 2019:\n"
            "- Phase 1 (up to 8 epochs, LR 1e-3): backbone frozen, head trained.\n"
            "- Phase 2 (up to 25 epochs, LR 1e-5, AdamW weight decay 1e-4): top 120 backbone layers "
            "fine-tuned with BatchNorm frozen.\n"
            "Both phases stop early on validation loss. Validation-only pilot trials may change some of "
            "these settings before the final fit."
        ),
    },
    {
        "keys": ["grad-cam", "gradcam", "heatmap", "saliency", "explainability", "layer 2"],
        "reply": (
            "**Grad-CAM — Layer 2 Regional Explainability**\n\n"
            "Gradient-weighted Class Activation Mapping (Grad-CAM) computes a 2D saliency "
            "heatmap by backpropagating gradients through the final convolutional layer "
            "(`top_activation` in EfficientNetB3) using `tf.GradientTape`.\n\n"
            "The heatmap identifies **which anatomical regions drove the classification** "
            "— e.g. the optic disc, macular area, or peripheral microaneurysm clusters. "
            "This provides radiologist-level regional accountability for every prediction.\n\n"
            "The overlay uses a Jet colormap: red = highest activation, blue = lowest."
        ),
    },
    {
        "keys": ["unet", "u-net", "segmentation", "lesion", "layer 3", "mask"],
        "reply": (
            "**Auxiliary U-Net — Layer 3 Pixel-Level Lesion Segmentation**\n\n"
            "The U-Net predicts a pixel-level lesion probability map, which is visualized "
            "with highlighted candidate regions.\n\n"
            "In the notebook training workflow, manual pixel masks were unavailable, so "
            "pseudo-masks were generated from the green channel, "
            "Top-Hat and Black-Hat morphology, and Grad-CAM saliency above 0.35. These are "
            "synthetic training targets, not manual ground truth.\n\n"
            "The notebook trains the U-Net with a **Hybrid Soft Dice + BCE Loss**. In the "
            "deployed app, the loaded U-Net mask is additionally filtered at 0.35 and combined "
            "with a Grad-CAM threshold of 0.30. The result is visual support, not an independent diagnosis."
        ),
    },
    {
        "keys": ["governance", "safety gate", "flag", "threshold", "confidence", "override"],
        "reply": (
            "**GovernanceAgent — Active Clinical Safety Gate**\n\n"
            "Unlike passive warning banners, the `GovernanceAgent` **actively intercepts** "
            "the pipeline when model confidence falls below the configurable threshold "
            "(default: 70%).\n\n"
            "When triggered:\n"
            "- Automated treatment guidance is **withheld entirely**.\n"
            "- `flagged_for_review: True` is set.\n"
            "- The case is rerouted to **mandatory human ophthalmologist triage**.\n\n"
            "This mirrors clinical safety governance patterns in defense medical AI systems "
            "so that the system does not act on an uncertain result for ambiguous or out-of-distribution images."
        ),
    },
    {
        "keys": ["cbr", "case-based reasoning", "similar cases", "embedding", "retrieval", "nearest neighbor"],
        "reply": (
            "**Case-Based Reasoning (CBR) — Similar Case Retrieval**\n\n"
            "After diagnosis, RetinaTrace extracts a **256-dimensional feature vector** "
            "from the penultimate dense layer (`head_dense`) of the classifier.\n\n"
            "When a reference library is available, this embedding is compared "
            "against stored image embeddings in `embeddings.npz` using **cosine similarity**. "
            "The top-3 matching image files may be displayed with their dataset-provided DR labels. "
            "These are not independently verified clinical outcomes, and similarity does not establish "
            "pathological equivalence.\n\n"
            "This is an exploratory case-comparison aid, not evidence that retrieval improves diagnosis."
        ),
    },
    {
        "keys": ["ben graham", "preprocessing", "preprocessing pipeline", "crop", "normalization"],
        "reply": (
            "**Preprocessing Pipeline — 4-Step Optical Standardization**\n\n"
            "1. **Circular Border Crop:** Strips non-informative black optical borders "
            "using luminance thresholding (green channel > 7).\n"
            "2. **Ben Graham Enhancement:** Applies spatial frequency subtraction:\n"
            "   `I_norm = 4×I − 4×GaussianBlur(I, σ=10) + 128`\n"
            "   This suppresses low-frequency illumination variation and enhances local detail, "
            "but may amplify noise in poor-quality images.\n"
            f"3. **Resize:** Images are resized to the model input size ({AppConfig.IMG_SIZE}×{AppConfig.IMG_SIZE} in this app).\n"
            "4. **Normalization:** Pixel values scaled to [0, 1] float32."
        ),
    },
    {
        "keys": ["kappa", "qwk", "quadratic weighted kappa", "metric", "evaluation"],
        "reply": (
            "**Quadratic Weighted Kappa (QWK) — Ordinal Evaluation Metric**\n\n"
            "Standard accuracy treats all misclassifications equally. In clinical DR grading, "
            "confusing Stage 0 (No DR) with Stage 4 (Proliferative) is catastrophically "
            "worse than confusing Stage 1 with Stage 2.\n\n"
            "QWK assigns **quadratic penalty weights** proportional to the distance between "
            "the predicted and true stage. A kappa of 1.0 = perfect agreement; "
            "0.0 = chance agreement; <0 = worse than chance.\n\n"
            + _REPORTED_METRICS_TEXT
        ),
    },
    {
        "keys": ["refer", "referral", "when to refer", "specialist", "urgency"],
        "reply": (
            "**Clinical Referral Guidelines (AAO PPP (Flaxel et al., 2020))**\n\n"
            "| DR Stage | Referral Urgency | Recall Interval |\n"
            "|---|---|---|\n"
            "| Stage 0 — No DR | No referral needed | 12 months |\n"
            "| Stage 1 — Mild NPDR | No immediate referral | 12 months |\n"
            "| Stage 2 — Moderate NPDR | Ophthalmologist referral recommended | 6–12 months |\n"
            "| Stage 3 — Severe NPDR | Retinal specialist referral | 3–4 months |\n"
            "| Stage 4 — Proliferative DR | **Urgent referral within 1–2 weeks** | ASAP |\n\n"
            "⚠️ *This tool is assistive clinical decision support only. All referral "
            "decisions must be confirmed by a qualified ophthalmologist.*"
        ),
    },
    {
        "keys": ["dataset", "data", "kaggle", "aptos", "idrid", "messidor", "eyepacs", "3662", "3,662"],
        "reply": (
            "**Dataset — APTOS 2019 Blindness Detection**\n\n"
            "The notebook trains on the **APTOS 2019 Blindness Detection** Kaggle competition data: the "
            "**3,662 labelled images** in `train_images/`, graded 0–4 by clinicians at Aravind Eye Hospital, "
            "India. The published class counts are 1,805 No DR, 370 Mild, 999 Moderate, 193 Severe and "
            "295 Proliferative (about 9.4× imbalance).\n\n"
            "Near-duplicate photographs are grouped with perceptual hashing, and whole groups are assigned "
            "to a 70/15/15 train/validation/test split. The data comes from a single source and has no "
            "patient IDs, so results may not transfer to other clinics or cameras. The checkpoint loaded in "
            "this app may predate the APTOS retraining; treat its predictions as a prototype demonstration."
        ),
    },
]

_CHATBOT_FALLBACK = (
    "I don't have a specific answer for that query in my clinical knowledge base. "
    "For questions about diabetic retinopathy management, please consult the "
    "**AAO Preferred Practice Pattern (Flaxel et al., 2020)** or a qualified ophthalmologist.\n\n"
    "You can ask me about: DR stages (0–4), Grad-CAM, the U-Net segmentation, "
    "the Governance Agent, Case-Based Reasoning, the preprocessing pipeline, "
    "Quadratic Weighted Kappa, referral guidelines, or the training dataset."
)

_SIMPLE_CHAT_REPLIES = {
    "lesion": (
        "A retinal lesion is an area of damage or abnormality in the retina, such as a "
        "microaneurysm, haemorrhage, or exudate. RetinaTrace highlights possible lesion areas "
        "to help a clinician review the image; the highlights are not a confirmed diagnosis."
    ),
    "grad-cam": (
        "Grad-CAM is a heatmap showing which parts of the retinal image influenced the model's "
        "prediction. It helps the user see where the model was looking."
    ),
    "unet": (
        "U-Net is the part of RetinaTrace that highlights possible lesion areas pixel by pixel. "
        "It helps show where abnormalities may be present, but the highlighted areas still need clinical confirmation."
    ),
    "efficientnet": (
        "EfficientNetB3 is the image-classification model used by RetinaTrace. It examines the "
        "retinal photograph and estimates the most likely diabetic-retinopathy stage."
    ),
    "governance": (
        "The Governance Agent is a safety check. If the result is uncertain or the supporting "
        "evidence does not agree, it withholds automated advice and asks for specialist review."
    ),
    "refer": (
        "Referral urgency depends on the detected retinopathy stage. More severe or uncertain "
        "results need faster review by an ophthalmologist."
    ),
    "stage 0": "Stage 0 means no diabetic-retinopathy signs were detected in the image.",
    "stage 1": "Stage 1 means mild diabetic retinopathy, usually limited to microaneurysms.",
    "stage 2": "Stage 2 means moderate diabetic retinopathy with more retinal changes than Stage 1.",
    "stage 3": "Stage 3 means severe non-proliferative diabetic retinopathy and needs prompt specialist review.",
    "stage 4": "Stage 4 means proliferative diabetic retinopathy, which requires urgent specialist assessment.",
}


class ChatRouterAgent:
    """Routes a chat message to a safe, explicit response path."""

    _COMMON_INTENTS = {
        "greeting": {"hi", "hii", "hiii", "hello", "helloo", "hey", "heyy", "goodmorning", "goodafternoon", "goodevening"},
        "help": {"help", "whatcanyoudo", "whatcaniask", "whatquestionscaniask"},
        "thanks": {"thanks", "thankyou", "thanku", "thankyouu", "thnks", "thx", "ty", "tq", "okay", "ok", "okk"},
        "definition": {"whatisdiabeticretinopathy", "whatisdr", "definediabeticretinopathy"},
    }

    @staticmethod
    def _technical_request(query: str) -> bool:
        technical_terms = [
            "student", "developer", "technical", "implementation", "architecture", "algorithm",
            "code", "training", "trained", "loss", "pipeline", "threshold", "model layers",
            "how does", "how is", "in detail", "under the hood", "show me",
        ]
        return any(term in query for term in technical_terms)

    @staticmethod
    def _close_match(query_key: str, phrases: set) -> bool:
        # Fuzzy matching is for typos only: a much longer question (e.g. "what is the 4-2-1 rule in
        # diabetic retinopathy") must not be mistaken for a short phrase it happens to contain.
        return any(
            query_key == phrase
            or (abs(len(query_key) - len(phrase)) <= 3
                and SequenceMatcher(None, query_key, phrase).ratio() >= 0.78)
            for phrase in phrases
        )

    def process(
        self,
        query: str,
        pred_context: Optional[dict] = None,
        history: Optional[List] = None,
    ) -> Dict[str, Any]:
        query_key = "".join(ch for ch in query if ch.isalnum())
        for intent, phrases in self._COMMON_INTENTS.items():
            if self._close_match(query_key, phrases):
                return {"intent": intent, "entry": None}

        high_risk_terms = [
            "should i start treatment", "should i take medicine", "what medication should i take",
            "can i start treatment", "should i inject", "prescribe", "dosage", "treat myself",
            # medicine and dosing questions, e.g. "What dose of insulin should I take?"
            "dose", "insulin", "metformin", "how much should i take", "should i take", "should i stop taking",
        ]
        if any(term in query for term in high_risk_terms):
            return {"intent": "high_risk", "entry": None}

        if pred_context and pred_context.get("stage_name"):
            urgent_words = ["urgent", "urgency", "red flag", "red-flag", "emergency", "same-day", "same day", "symptom"]
            if pred_context.get("urgent") and any(word in query for word in urgent_words):
                return {"intent": "prediction_context", "entry": None, "topic": "red_flag"}
            if query_key in {"why", "whythis", "explain", "more", "moreinfo", "whataboutthat"}:
                return {"intent": "prediction_context", "entry": None, "topic": "classification"}
            triggers = ["why", "classified", "this image", "current", "result", "prediction", "explain this",
                        "confidence", "how confident", "what was found", "lesion", "quadrant", "peak",
                        "this patient", "this case", "referred", "refer this"]
            if any(trigger in query for trigger in triggers):
                if any(word in query for word in ["lesion", "exudate", "microaneurysm"]):
                    topic = "lesions"
                elif any(word in query for word in ["confidence", "certain"]):
                    topic = "confidence"
                elif any(word in query for word in ["urgent", "urgency", "follow-up", "follow up", "action", "refer"]):
                    topic = "advisory"
                elif any(word in query for word in ["quadrant", "peak", "grad-cam", "gradcam"]):
                    topic = "attention"
                else:
                    topic = "classification"
                return {"intent": "prediction_context", "entry": None, "topic": topic}

        for entry in _CLINICAL_KB:
            if any(keyword in query for keyword in entry["keys"]):
                return {
                    "intent": "knowledge_base",
                    "entry": entry,
                    "technical": self._technical_request(query),
                }
        return {"intent": "unknown", "entry": None}


class ChatKnowledgeAgent:
    """Builds a response from the routed knowledge or approved prediction context."""

    def process(self, query: str, route: Dict[str, Any], pred_context: Optional[dict] = None) -> str:
        intent = route["intent"]
        if intent == "greeting":
            return "Hello. I am the **RetinaTrace Clinical Knowledge Assistant**. Ask me about diabetic retinopathy stages, this model, or clinical guidelines."
        if intent == "help":
            return "I can explain **DR stages 0–4**, Grad-CAM, U-Net lesion segmentation, the Governance Agent, referral guidance, preprocessing, and the current prediction."
        if intent == "thanks":
            return "You are welcome. Ask another question whenever you are ready."
        if intent == "definition":
            return "**Diabetic retinopathy** is damage to the retinal blood vessels caused by diabetes. It progresses from no retinopathy through non-proliferative stages to proliferative disease. Regular eye screening and good blood-glucose and blood-pressure control are important."
        if intent == "high_risk":
            return (
                "I cannot recommend starting, stopping, or changing treatment from an AI chat response. "
                "Please discuss medication or procedures with a qualified ophthalmologist or the patient's clinician."
            )
        if intent == "knowledge_base":
            if not route.get("technical"):
                for keyword, simple_reply in _SIMPLE_CHAT_REPLIES.items():
                    if keyword in query:
                        return simple_reply
            return route["entry"]["reply"]
        if intent == "prediction_context":
            stage_name = pred_context.get("stage_name", "Unknown")
            conf = pred_context.get("confidence", 0.0)
            topic = route.get("topic", "classification")
            if topic == "red_flag":
                symptoms = "\n".join(f"- {flag}" for flag in pred_context.get("red_flags", []))
                return (
                    f"**Why this result is marked {pred_context.get('outcome', 'URGENT')}**\n\n"
                    f"The patient reported these red-flag symptoms:\n{symptoms}\n\n"
                    "These symptoms can signal retinal detachment, vitreous haemorrhage or another emergency "
                    "that a photograph cannot rule out, so the case needs same-day eye care whatever the image shows. "
                    f"The image itself was graded **{stage_name}** ({conf*100:.1f}% confidence); the red flags do not "
                    "change that grade, they only raise the urgency. Automated treatment advice is withheld."
                )
            if topic == "lesions":
                return (
                    f"**Lesions in the current analysis**\n\n"
                    f"The U-Net identified lesion candidates in **{pred_context.get('lesion_pct', 0.0):.1f}%** of the retinal area. "
                    "These candidates may include microaneurysms or exudate-like regions and should be clinically confirmed."
                )
            if topic == "confidence":
                return f"The current **{stage_name}** prediction has a model confidence of **{conf*100:.1f}%**."
            if topic == "advisory":
                urgency = pred_context.get("urgency", "N/A")
                followup = pred_context.get("followup", "N/A")
                plan = pred_context.get("plan", "")
                if pred_context.get("urgent") or "HUMAN SPECIALIST TRIAGE" in urgency:
                    return (
                        "This result should be reviewed by an ophthalmologist before any action. "
                        f"{pred_context.get('governance_message', '')} "
                        "Automated treatment advice should not be followed without clinical confirmation."
                    )
                return (
                    f"**Current clinical routing**\n\n"
                    f"- Urgency: **{urgency}**\n"
                    f"- Follow-up: **{followup}**\n"
                    f"- Action: {plan}"
                )
            if topic == "attention":
                return (
                    f"Grad-CAM identified **{pred_context.get('peak_quadrant', '-')}** as the peak attention quadrant. "
                    "This shows which region influenced the model and is not, by itself, a diagnosis."
                )
            if topic == "classification":
                return (
                    f"The image was classified as **{stage_name}** with **{conf*100:.1f}% confidence**. "
                    f"The main supporting signals were the Grad-CAM peak in **{pred_context.get('peak_quadrant', '-')}** "
                    f"and U-Net lesion candidates covering **{pred_context.get('lesion_pct', 0.0):.1f}%** of the retinal area. "
                    "Because this result is AI-assisted, it should be confirmed by an ophthalmologist."
                )
            return (
                f"**Current Prediction Context: {stage_name}**\n\n"
                f"The model classified this fundus image as **{stage_name}** with a confidence of **{conf*100:.1f}%**.\n\n"
                f"- Grad-CAM peak quadrant: **{pred_context.get('peak_quadrant', '-')}**\n"
                f"- U-Net lesion candidates: **{pred_context.get('lesion_pct', 0.0):.1f}%**\n"
                f"- Urgency: **{pred_context.get('urgency', 'N/A')}**\n"
                f"- Follow-up: **{pred_context.get('followup', 'N/A')}**\n"
                f"- Action: {pred_context.get('plan', 'N/A')}"
            )
        return _CHATBOT_FALLBACK


class ChatGovernanceAgent:
    """Applies a final clinical disclaimer before a chat response is released."""

    def process(self, reply: str, route: Dict[str, Any]) -> str:
        if route["intent"] in {"knowledge_base", "prediction_context", "definition"} and "qualified ophthalmologist" not in reply:
            return reply + "\n\n*This is AI-assisted information. Confirm clinical decisions with a qualified ophthalmologist.*"
        return reply


def run_chat_pipeline(
    message: str,
    pred_context: Optional[dict] = None,
    history: Optional[List] = None,
) -> str:
    """Run the chatbot router, knowledge/context agent, and final safety gate."""
    query = str(message).lower().strip()
    router = ChatRouterAgent()
    route = router.process(query, pred_context, history)
    response = ChatKnowledgeAgent().process(query, route, pred_context)
    if route["intent"] == "prediction_context" and pred_context and pred_context.get("demo_preset"):
        response = (
            "⚠️ *The current result is a demo preset with fixed illustrative values, "
            "not a model prediction.*\n\n" + response
        )
    return ChatGovernanceAgent().process(response, route)


def respond_to_clinical_query(message: str, history: Optional[List] = None, pred_context: dict = None) -> tuple:
    """Gradio adapter for the explicit multi-agent chat pipeline."""
    history = [] if history is None else list(history)
    if not message or not str(message).strip():
        return history, ""
    reply = run_chat_pipeline(message, pred_context, history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, ""
