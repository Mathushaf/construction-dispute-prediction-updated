import os
import sys
from flask import Flask, render_template, request

# ======================================================
# Ensure parent folder is accessible for model imports
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# ======================================================
# Import ONLY the two models used in the paper
# ======================================================
from DisputeIndex.DisputeIndexUI import predict_index
from DisputeClassificationModel.DisputeclassifierUI import predict_dispute

# ======================================================
# Initialize Flask
# ======================================================
app = Flask(__name__)

# ======================================================
# Dispute Index Mapping
# ======================================================
def get_index_color(label: int) -> dict:
    """
    Maps the Dispute Index class (1-5)
    to a level name and description.
    """

    mapping = {
        1: ("Very Low", "Minimal likelihood of dispute"),
        2: ("Low", "Low likelihood of dispute"),
        3: ("Moderate", "Moderate likelihood of dispute"),
        4: ("High", "High likelihood of dispute"),
        5: ("Very High", "Very strong likelihood of dispute"),
    }

    if label in mapping:
        return {
            "color": mapping[label][0],
            "description": mapping[label][1]
        }

    return {
        "color": "Unknown",
        "description": "Index out of expected range (1-5)"
    }


# ======================================================
# Incident Separation
# ======================================================
def split_incidents(text: str):
    """
    Separates multiple incidents using:
    ===NEW INCIDENT===

    If the marker is not present,
    the whole text is treated as one incident.
    """

    incidents = [
        incident.strip()
        for incident in text.split("===NEW INCIDENT===")
        if incident.strip()
    ]

    return incidents


# ======================================================
# Extract Dispute Index
# ======================================================
def extract_index_label(index_result):
    """
    Safely extracts the predicted Dispute Index label.
    Expected model output:
    {"predicted_label": 1-5}
    """

    if (
        isinstance(index_result, dict)
        and "predicted_label" in index_result
    ):
        try:
            return int(index_result["predicted_label"])
        except (TypeError, ValueError):
            return 0

    return 0


# ======================================================
# Main Flask Route
# ======================================================
@app.route("/", methods=["GET", "POST"])
def home():

    # --------------------------------------------------
    # Email form values
    # --------------------------------------------------
    email = {
        "category": "",
        "subject": "",
        "to": "",
        "description": ""
    }

    # --------------------------------------------------
    # Analysis output
    #
    # Existing fields such as score and level are kept
    # so the gauge in index.html continues to work.
    # --------------------------------------------------
    analysis = {
        "score": None,
        "level": None,
        "level_description": None,
        "classification": "Not analysed yet",
        "description_highlighted": None,

        # Multi-incident results
        "incidents": []
    }

    # ==================================================
    # POST REQUEST
    # ==================================================
    if request.method == "POST":

        # ----------------------------------------------
        # Read input fields
        # ----------------------------------------------
        to_text = request.form.get("to", "").strip()
        subject_text = request.form.get("subject", "").strip()
        description_text = request.form.get(
            "description",
            ""
        ).strip()

        # ----------------------------------------------
        # Keep entered email values after prediction
        # ----------------------------------------------
        email.update({
            "to": to_text,
            "subject": subject_text,
            "description": description_text
        })

        # ==============================================
        # Run analysis only if description exists
        # ==============================================
        if description_text:

            # ------------------------------------------
            # STEP 1:
            # Split full communication into incidents
            # ------------------------------------------
            incidents = split_incidents(
                description_text
            )

            incident_results = []

            # ------------------------------------------
            # STEP 2:
            # Analyse every incident separately
            # ------------------------------------------
            for incident_number, incident in enumerate(
                incidents,
                start=1
            ):

                # ======================================
                # Dispute Index Prediction
                # ======================================
                index_result = predict_index(
                    incident
                )

                index_label = extract_index_label(
                    index_result
                )

                index_info = get_index_color(
                    index_label
                )

                # ======================================
                # Dispute Type Prediction
                # ======================================
                classification_result = predict_dispute(
                    incident
                )

                # ======================================
                # Store result for this incident
                # ======================================
                incident_results.append({

                    "incident_number":
                        incident_number,

                    "text":
                        incident,

                    "score":
                        index_label,

                    "level":
                        index_info["color"],

                    "level_description":
                        index_info["description"],

                    "classification":
                        classification_result
                })

            # ==========================================
            # STEP 3:
            # Calculate Overall Dispute Index
            #
            # Highest incident score controls gauge.
            # ==========================================
            if incident_results:

                valid_scores = [
                    item["score"]
                    for item in incident_results
                    if item["score"] > 0
                ]

                if valid_scores:

                    overall_score = max(
                        valid_scores
                    )

                    overall_info = get_index_color(
                        overall_score
                    )

                    overall_level = (
                        overall_info["color"]
                    )

                    overall_description = (
                        overall_info["description"]
                    )

                else:

                    overall_score = 0
                    overall_level = "Unknown"
                    overall_description = (
                        "No valid Dispute Index prediction"
                    )

            else:

                overall_score = 0
                overall_level = "Unknown"
                overall_description = (
                    "No valid incident detected"
                )

            # ==========================================
            # STEP 4:
            # Update analysis object
            #
            # score = highest incident index
            # incidents = all incident-level outputs
            # ==========================================
            analysis.update({

                "score":
                    overall_score,

                "level":
                    overall_level,

                "level_description":
                    overall_description,

                # Keep this for compatibility,
                # but incident-level classifications
                # are shown in the table.
                "classification":
                    "See incident-level results",

                "description_highlighted":
                    description_text.replace(
                        "\n",
                        "<br>"
                    ),

                "incidents":
                    incident_results
            })

    # ==================================================
    # Render page
    # ==================================================
    return render_template(
        "index.html",
        email=email,
        analysis=analysis
    )


# ======================================================
# Run Flask Application
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
