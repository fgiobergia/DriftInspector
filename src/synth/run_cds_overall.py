from tqdm import tqdm

from wrappers import (
    AgrawalWrapper,
    SEAWrapper,
    LEDWrapper,
    HyperplaneWrapper,
    STAGGERWrapper,
)
from skmultiflow.data import ConceptDriftStream

from argparse import ArgumentParser

from sklearn.tree import DecisionTreeClassifier

import pickle

import pathlib


def train_and_drift_overall_start_end(
    DataSource,
    drift=True,
    ds_kwargs={},
    train_size=5_000,
    n_batches=50,
    batch_size=200,
    ClfModel=DecisionTreeClassifier,
    clf_kwargs={},
    overall_detectors_args={},
    batch_idx_drift_start=45,
):
    """Train a classifier on a stream and detect drift using multiple detectors
    args:
        DataSource: class that generates the data stream
        drift: whether to include drift in the data stream
        ds_kwargs: arguments to pass to the DataSource class
        train_size: number of samples to use for training
        n_batches: number of batches
        batch_size: size of each batch
        ClfModel: classifier model to use
        clf_kwargs: arguments to pass to the classifier
        overall_detectors_args: arguments to pass to the drift detectors
    returns:
        overall_drift_result: dictionary containing the results of the drift detection
    """

    # Import drift detectors
    from skmultiflow.drift_detection.hddm_a import HDDM_A
    from skmultiflow.drift_detection.eddm import EDDM
    from skmultiflow.drift_detection import DDM
    from skmultiflow.drift_detection.adwin import ADWIN
    from alibi_detect.cd import ChiSquareDrift, FETDrift
    from skmultiflow.drift_detection import KSWIN
    from skmultiflow.drift_detection import PageHinkley

    if overall_detectors_args == {}:
        # Set default values for the drift detectors
        overall_detectors_args = {
            "hddma_drift_confidence": [0.0001, 0.0002, 0.001, 0.002, 0.01, 0.02],
            "ddm_min_num_instances": [5, 10, 30, 50, 100, batch_size],
            "adwin_delta": [0.0001, 0.0002, 0.001, 0.002, 0.01, 0.02],
            "eddm": None,
            "chi2": [0.001, 0.01, 0.25, 0.05],  # pvalue
            "fet": [0.001, 0.01, 0.25, 0.05],  # pvalue
            "kswin_window_size": [50, 100, batch_size, 300, 400, 500, 600, 700, 800],
            "kswin_alpha": 0.0001,
            "pagehinkley_min_num_instances": [5, 10, 30, 50, 100, batch_size],
        }

    tot_samples = n_batches * batch_size  # total number of samples
    position = (n_batches // 2) * batch_size  # "center" of the drift
    width = (n_batches // 4) * batch_size  # size of the transitory

    data_source = DataSource(**ds_kwargs)
    stream = data_source.stream
    drift_stream = data_source.drift_stream

    if drift:
        # If drift is True, we use a different stream for the drift, i.e., the drift_stream
        cds = ConceptDriftStream(
            stream=stream, drift_stream=drift_stream, position=position, width=width
        )
    else:
        # If drift is False, we use the same stream for both
        cds = ConceptDriftStream(
            stream=stream, drift_stream=stream, position=position, width=width
        )  # no drift!

    X_train, y_train = stream.next_sample(train_size)

    clf = ClfModel(**clf_kwargs)
    clf.fit(X_train, y_train)

    detectors_dict = {}

    params = {}
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
            # kswin_i = KSWIN(window_size=kswin_window, stat_size=int(kswin_window / 3))
            alpha = overall_detectors_args["kswin_alpha"]
            kswin_i = KSWIN(window_size=kswin_window, stat_size=30, alpha=alpha)
            detectors_dict[f"kswin_{kswin_window}_{alpha}"] = kswin_i

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
        for chi2_params in overall_detectors_args["chi2"]:
            detectors_dict[f"chi2_{chi2_params}"] = chi2_params
            params[f"chi2_{chi2_params}"] = chi2_params

    if "fet" in overall_detectors_args:
        # Fisher Exact Test
        for fet_params in overall_detectors_args["fet"]:
            detectors_dict[f"fet_{fet_params}"] = fet_params
            params[f"fet_{fet_params}"] = fet_params

    # y_pred_train = clf.predict(X_train).astype(int)

    # hddm_a, eddm: Whether the last sample analyzed was correctly classified or not. 1 indicates an error (miss-classification).
    # errors_train = (y_train.astype(int) != y_pred_train).astype(int)

    # ADWIN: 0: Means the learners prediction was wrong, 1: Means the learners prediction was correct
    # corrects_train = (y_train.astype(int) == y_pred_train).astype(int)

    """
    # Add the training data to the drift detector
    # We initialize the detectors with the training data
    for i in range(len(errors_train)):
        for detector_name, detector in zip(detector_names, detectors):
            if detector_name[0:5] == "adwin":
                # ADWIN use 1 for correct predictions, 0 for wrong predictions
                detector.add_element(corrects_train[i])
            else:
                # The others (hddm_a and eddm) use 1 for a wrong predictions, 0 for correct
                detector.add_element(errors_train[i])

    """

    # Initialize dictionaries to store the results
    detector_warnings = {detector_name: {} for detector_name in detectors_dict}
    detector_detected = {detector_name: {} for detector_name in detectors_dict}
    overall_drift_result = {detector_name: {} for detector_name in detectors_dict}

    for batch_idx in range(n_batches):
        X_batch, y_batch = cds.next_sample(batch_size)

        y_pred = clf.predict(X_batch)

        # hddm_a, eddm: Whether the last sample analyzed was correctly classified or not. 1 indicates an error (miss-classification).
        errors_b = (y_batch.astype(int) != y_pred).astype(int)

        # ADWIN: 0: Means the learners prediction was wrong, 1: Means the learners prediction was correct
        corrects_b = (y_batch.astype(int) == y_pred).astype(int)

        # We dot this oly for  "chi2" or "fet"  detectors:
        if batch_idx < 5:
            # We initialize chi and fet for the entire batch
            # overall_detectors_args['chi2'] and overall_detectors_args['fet']  is the p-value threshold for the chi-square test
            for detector_name, detector in detectors_dict.items():
                if detector_name[0:4] == "chi2" or detector_name[0:3] == "fet":
                    if type(detector) == float:
                        # We initialize chi and fet for the entire batch
                        if detector_name[0:4] == "chi2":
                            chi = ChiSquareDrift(errors_b, params[detector_name])
                            detectors_dict[detector_name] = chi
                        if detector_name[0:3] == "fet":
                            fet = FETDrift(errors_b, params[detector_name])
                            detectors_dict[detector_name] = fet
        else:
            if batch_idx >= batch_idx_drift_start:
                for detector_name, detector in detectors_dict.items():
                    if detector_name[0:4] == "chi2" or detector_name[0:3] == "fet":
                        # We evaluate chi or fet for the entire batch
                        preds = detectors_dict[detector_name].predict(errors_b)
                        is_drift = preds["data"]["is_drift"]
                        if is_drift:
                            # Add detected change to the dictionary
                            # We say that it detect a drift for all samples in the batch
                            detector_detected[detector_name][batch_idx] = [
                                1 for i in range(batch_size)
                            ]

        # For the other approaches, we iterate one sample at the time
        for i in range(len(errors_b)):
            for detector_name, detector in detectors_dict.items():

                if batch_idx < 5 or batch_idx >= batch_idx_drift_start:
                    if detector_name[0:4] == "chi2" or detector_name[0:3] == "fet":
                        # We skip chi2 and fet has we do the evaluation for the entire batch
                        continue
                    elif detector_name[0:5] == "adwin":
                        detector.add_element(corrects_b[i])
                    else:
                        detector.add_element(errors_b[i])

                # Start detecting change after the first batch
                if batch_idx >= batch_idx_drift_start:

                    if detector.detected_warning_zone():
                        # Add warning zone to the dictionary

                        if batch_idx not in detector_warnings[detector_name]:
                            detector_warnings[detector_name][batch_idx] = []
                        detector_warnings[detector_name][batch_idx].append(i)
                        # print('Warning zone has been detected in data: ' + str(errors_b[i]) + ' - of index: ' + str(i))
                    if detector.detected_change():
                        # Add detected change to the dictionary
                        # print(f"{detector_name} - Change has been detected in batch_idx: {batch_idx} - of index: {i}")
                        if batch_idx not in detector_detected[detector_name]:
                            detector_detected[detector_name][batch_idx] = []
                        detector_detected[detector_name][batch_idx].append(i)

    # Store subgroup results
    for detector_name in detector_detected:
        overall_drift_result[detector_name] = {
            # "warnings": detector_warnings[detector_name],
            "detected_batch": detector_detected[detector_name],
            "num_pts_detected": sum(
                [len(v) for v in detector_detected[detector_name].values()]
            ),
        }

    return overall_drift_result


if __name__ == "__main__":

    parser = ArgumentParser(description="Compute dataset statistics")

    parser.add_argument(
        "--exp_type",
        help="Experiment type",
        required=True,
        type=str,
        choices=["agrawal", "sea", "led", "hyper", "stagger"],
    )

    parser.add_argument(
        "--n_exp", help="Number of experiments", default=100, required=False, type=int
    )

    parser.add_argument(
        "--train_size", help="Training size", default=5000, required=False, type=int
    )

    parser.add_argument(
        "--n_batches", help="Number of batches", default=50, required=False, type=int
    )

    parser.add_argument(
        "--batch_size", help="Batch size", default=200, required=False, type=int
    )

    parser.add_argument(
        "--noise", help="Noise percentage", default=0.1, required=False, type=float
    )
    parser.add_argument(
        "--output_dir_name",
        help="Directory where the results are stored",
        default="results-5/results-overall-drift-datasets",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--batch_idx_drift_start",
        help="Batch index where drift starts",
        default=45,
        required=False,
        type=int,
    )

    args = parser.parse_args()

    exp_type = args.exp_type
    n_exp = args.n_exp
    train_size = args.train_size
    n_batches = args.n_batches
    batch_size = args.batch_size
    noise = args.noise
    batch_idx_drift_start = args.batch_idx_drift_start
    output_dir_name = (
        f"{args.output_dir_name}-noise-{noise}_start_{batch_idx_drift_start}"
    )

    print("Output directory: ", output_dir_name)
    print("Experiment type: ", exp_type)

    overall_drift_results = {}
    gt = []

    if exp_type == "agrawal":
        DataClass = AgrawalWrapper
        data_kwargs = {"perturbation": noise}
    elif exp_type == "sea":
        DataClass = SEAWrapper
        data_kwargs = {"noise_percentage": noise}
    elif exp_type == "led":
        DataClass = LEDWrapper
        data_kwargs = {"noise_percentage": noise}
    elif exp_type == "stagger":
        DataClass = STAGGERWrapper
        data_kwargs = {}
    elif exp_type == "hyper":
        DataClass = HyperplaneWrapper
        data_kwargs = {"noise_percentage": noise}
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")

    i = 0
    # overall_detectors_args = {
    #    "kswin_window_size": [50, batch_size, 700],
    # }

    for drift in [False, True]:
        print(f"Drift: {drift}")
        for exp in tqdm(range(n_exp // 2)):  # positive samples

            overall_drift_result = train_and_drift_overall_start_end(
                DataClass,
                ds_kwargs={"random_state": i, **data_kwargs},
                drift=drift,
                train_size=train_size,
                n_batches=n_batches,
                batch_size=batch_size,
                ClfModel=DecisionTreeClassifier,
                clf_kwargs={},
                batch_idx_drift_start=batch_idx_drift_start,
                # overall_detectors_args=overall_detectors_args,
            )

            for method in overall_drift_result:
                if method not in overall_drift_results:
                    overall_drift_results[method] = []

                # Set to 1 if the experiment has drift, 0 otherwise
                overall_drift_result[method]["drift"] = int(drift)
                overall_drift_results[method].append(overall_drift_result[method])

                # print(exp, method, overall_drift_result[method])

            # gt[i] = drift
            gt.append(drift)

            i += 1

    pathlib.Path(output_dir_name).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir_name}/{exp_type}_drift_results_overall.pkl", "wb") as f:
        pickle.dump(overall_drift_results, f)


"""

0.2 vs 0.05
0.1 vs 0.025
python run_cds_overall.py --exp_type agrawal
python run_cds_overall.py --exp_type sea
python run_cds_overall.py --exp_type led
python run_cds_overall.py --exp_type hyper
python run_cds_overall.py --exp_type stagger

for noise in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    echo 
    python run_cds_overall.py --exp_type agrawal --noise $noise --output_dir_name results-chi-fet/results-overall-drift-datasets-chi-fet
    python run_cds_overall.py --exp_type sea --noise $noise --output_dir_name results-chi-fet/results-overall-drift-datasets-chi-fet
    python run_cds_overall.py --exp_type led --noise $noise --output_dir_name results-chi-fet/results-overall-drift-datasets-chi-fet
    python run_cds_overall.py --exp_type hyper --noise $noise --output_dir_name results-chi-fet/results-overall-drift-datasets-chi-fet
    python run_cds_overall.py --exp_type stagger --noise $noise --output_dir_name results-chi-fet/results-overall-drift-datasets-chi-fet
done
"""
