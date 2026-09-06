class DetectorEnsemble:

    def __init__(self, detectors, weights=None):
        self.detectors = detectors

        if weights is None:
            self.weights = {
                name: 1.0
                for name in detectors
            }
        else:
            self.weights = weights

    def combine(self, results):
        """
        Combine detector probabilities using weighted averaging.

        Each detector result must contain:
            spoof_score
            bonafide_score
        """

        total_weight = 0.0
        spoof_score = 0.0
        bonafide_score = 0.0

        for name, result in results.items():

            weight = self.weights.get(name, 1.0)

            spoof_score += (
                result["spoof_score"] * weight
            )

            bonafide_score += (
                result["bonafide_score"] * weight
            )

            total_weight += weight

        if total_weight == 0:
            raise ValueError("Total detector weight cannot be zero.")

        spoof_score /= total_weight
        bonafide_score /= total_weight

        prediction = (
            "SPOOF"
            if spoof_score > bonafide_score
            else "BONAFIDE"
        )

        return {
            "prediction": prediction,
            "spoof_score": spoof_score,
            "bonafide_score": bonafide_score,
        }
