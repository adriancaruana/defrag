DIRECT = "direct"
RELATED = "related"
UNRELATED = "unrelated"


GENERAL_RELEVANCY = {
    "16.35": DIRECT,# Cancer chemotherapy [A]
    "8.4": DIRECT,# Other therapeutic procedures; hemic and lymphatic system [C]

    "16.25": DIRECT,#  Therapeutic radiology [A]
    "16.38": RELATED,#  Other diagnostic procedures (interview; evaluation; consultation) [C]
    "16.42": DIRECT,#  Other therapeutic procedures [C]
    "16.2": RELATED,#  Computerized axial tomography (CT) scan [C]

}

MELANOMA_RELEVANCY = {
    "15.6": DIRECT,# Skin graft [C]
    "15.4": DIRECT,# Excision of skin lesion [C]
    **GENERAL_RELEVANCY,
}

LUNG_RELEVANCY = {
    "6.8": DIRECT,#  Other non-OR therapeutic procedures on respiratory system [B]
    "16.5": RELATED,# Routine chest X-ray [B]
    "6.4": DIRECT,# Diagnostic bronchoscopy and biopsy of bronchus [D]
    "6.3": DIRECT,#  Lobectomy or pneumonectomy [C]
    "6.9": DIRECT,#  Other OR therapeutic procedures on respiratory system [C]
    **GENERAL_RELEVANCY,
}
    
BREAST_RELEVANCY = {
    "15.9": DIRECT,# Other OR therapeutic procedures on skin and breast [A]
    "16.5": RELATED,# Routine chest X-ray [B]
    **GENERAL_RELEVANCY,
}

RELEVANCY_DICT = {
    "breast": BREAST_RELEVANCY,
    "lung": LUNG_RELEVANCY,
    "melanoma": MELANOMA_RELEVANCY,
}