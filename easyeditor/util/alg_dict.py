from ..dataset import FGVEditDataset
from ..models.ike import apply_ike_to_multimodal_model
from ..models.mend import MendMultimodalRewriteExecutor
from ..models.serac import SeracMultimodalRewriteExecutor


ALG_MULTIMODAL_DICT = {
    "IKE": apply_ike_to_multimodal_model,
    "MEND": MendMultimodalRewriteExecutor().apply_to_model,
    "SERAC": SeracMultimodalRewriteExecutor().apply_to_model,
}

MULTIMODAL_DS_DICT = {
    "fg": FGVEditDataset,
}
