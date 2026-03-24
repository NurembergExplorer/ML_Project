# Report Notes

## Helpful explanation
- High NDVI and low SWIR-proxy water stress features are plausibly useful for vegetation prediction.
- Ridge coefficients are useful for directionality under the current feature representation.
- LightGBM gain importance is useful for ranking which input features the model relied on most.

## Misleading explanation
- Feature importance is **not causal**.
- A high importance score for NDVI does not prove vegetation causes the prediction in a causal sense.
- Mixed cells, label aggregation, seasonal effects, and WorldCover noise can all make an explanation look convincing but still be misleading.

## Data issues
1. Low valid_frac cells likely reflect cloud, masking, or incomplete observations.
2. WorldCover labels are noisy, especially in transition zones and mixed-use urban cells.
3. 100 m grid cells aggregate multiple land uses, so composition targets may smooth real edges.
4. Temporal comparability is imperfect because yearly composites may still differ in acquisition mix and seasonal conditions.
5. The 'other' class groups heterogeneous land covers and is harder to interpret.

## One issue not fixed
- The project still inherits label uncertainty from ESA WorldCover and mixed-pixel aggregation. This is acknowledged but not fully solved.

## Arguing Against ChatGPT – Case 1
- Reject the suggestion to use CNNs or segmentation models.
- The assignment explicitly asks for tabular ML from satellite-derived features, not end-to-end image models.
- Using Ridge/LightGBM is both compliant and more defensible.

## Arguing Against ChatGPT – Case 2
- Reject the claim that a highlighted change tile guarantees real-world development.
- Predicted change is model-based and may reflect label noise, seasonal differences, or feature instability.
- High uncertainty or low valid_frac should weaken confidence in the interpretation.
