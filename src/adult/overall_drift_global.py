# Import drift detectors
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import KSWIN
from skmultiflow.drift_detection import PageHinkley


def read_experiment(tgt):
    import pickle
    import numpy as np

    # if os.path.exists(output_filename):
    #    continue

    subgroup_config_name = tgt.split("target-")[1].replace(".pkl", "")

    try:
        with open(tgt, "rb") as f:
            obj = pickle.load(f)
            sg = frozenset(obj["subgroup"])
            y_trues = obj["y_trues"]
            y_preds = obj["y_preds"]
            altered = obj["altered"]
            altered = [np.array(a) for a in altered]

            return {
                "altered": altered,
                "subgroup_config_name": subgroup_config_name,
                "sg": sg,
                "y_trues": y_trues,
                "y_preds": y_preds,
            }

    except pickle.UnpicklingError:
        print("Error file: " + tgt)
        return None

    except EOFError:
        print("Error file: " + tgt)
        return None


def init_detectors(overall_detectors_args=None):
    """Initialize the drift detectors
    Args:
        overall_detectors_args: dictionary with the parameters for the drift detectors
    Returns:
        detectors_dict: dictionary with the drift detectors - key: detector_name, value: detector_object

    """
    detectors_dict = {}
    if overall_detectors_args == {}:
        # Set default values for the drift detectors
        overall_detectors_args = {
            "hddma_drift_confidence": [0.0001, 0.0002, 0.001, 0.002, 0.01, 0.02],
            "ddm_min_num_instances": [500, 1000, 2000, 4000, 8000],
            "adwin_delta": [0.0001, 0.0002, 0.001, 0.002],
            "eddm": None,
            "chi2": 0.05,  # pvalue
            "fet": 0.05,  # pvalue
            "kswin_window_size": [500, 1000, 2000, 4000, 8000],
            "pagehinkley_min_num_instances": [500, 1000, 2000, 4000, 8000],
        }

    # Initialize drift detectors
    if "hddma_drift_confidence" in overall_detectors_args:
        for hddma_params in overall_detectors_args["hddma_drift_confidence"]:
            hddma_i = HDDM_A(drift_confidence=hddma_params)
            detectors_dict[f"hddma_{hddma_params}"] = hddma_i

    if "ddm_min_num_instances" in overall_detectors_args:
        for ddm_params in overall_detectors_args["ddm_min_num_instances"]:
            ddm_i = DDM(min_num_instances=ddm_params)
            detectors_dict[f"ddm_{ddm_params}"] = ddm_i

    if "adwin_delta" in overall_detectors_args:
        for adwin_params in overall_detectors_args["adwin_delta"]:
            adwin_i = ADWIN(adwin_params)
            detectors_dict[f"adwin_{adwin_params}"] = adwin_i

    if "kswin_window_size" in overall_detectors_args:
        for kswin_window in overall_detectors_args["kswin_window_size"]:
            # We leave the alpha as default (0.005)
            kswin_i = KSWIN(window_size=kswin_window, stat_size=int(kswin_window / 3))
            detectors_dict[f"kswin_{kswin_window}"] = kswin_i

    if "pagehinkley_min_num_instances" in overall_detectors_args:
        for pagehinkley_params in overall_detectors_args[
            "pagehinkley_min_num_instances"
        ]:
            pagehinkley_i = PageHinkley(min_instances=pagehinkley_params)
            detectors_dict[f"pagehinkley_{pagehinkley_params}"] = pagehinkley_i

    if "eddm" in overall_detectors_args:
        eddm = EDDM()
        detectors_dict["eddm"] = eddm

    if "chi2" in overall_detectors_args:
        detectors_dict["chi2"] = overall_detectors_args["chi2"]

    if "fet" in overall_detectors_args:
        # Fisher Exact Test
        detectors_dict["fet"] = overall_detectors_args["fet"]

    return detectors_dict, overall_detectors_args


def get_cm_detections(altered_sg_batch, method_warnings):
    """
    Returns the confusion matrix for the detections of a method
    Args:
        altered_sg_batch: list of altered subgroups per batch
        method_warnings: dict of batch_idx: list of warnings or detections
    Returns:
        tp, fp, fn, tn

    """
    tp, fp, fn, tn = 0, 0, 0, 0

    for batch_idx in range(1, len(altered_sg_batch)):
        num_altered = altered_sg_batch[batch_idx]
        if batch_idx in method_warnings:
            num_warnings = len(method_warnings[batch_idx])
        else:
            num_warnings = 0

        # Altered and detected
        if num_altered > 0 and num_warnings > 0:
            tp += 1
        elif num_altered == 0 and num_warnings > 0:
            fp += 1
        elif num_altered > 0 and num_warnings == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn
