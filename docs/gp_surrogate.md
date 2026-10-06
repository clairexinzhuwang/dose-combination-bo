# How the model predicts responses

The model uses the efficacy and toxicity observed so far to estimate mean
responses at every dose combination, including combinations not yet tried.
It also shows how uncertain these estimates are. Responses at nearby
combinations are treated as related, so an observation can inform predictions
at other doses. Predictions update after each group of participants.
All four rules use the same models.

## Technical details

### Model and fitting

Efficacy and toxicity have separate Gaussian-process (GP) models. Each has a
constant mean, scaled Matérn-5/2 kernel, one lengthscale per dose axis and
Gaussian observation noise.

After each observed cohort, the mean, lengthscales and output scale are fitted
with 120 Adam steps at learning rate 0.1. cKG holds these fitted parameters
fixed during each hypothetical update. Gaussian conditioning is exact given
those parameters; fitting is not guaranteed to maximize the marginal likelihood.

### Observation variance

The primary study uses the data-generating outcome standard deviations (SDs)
as fixed inputs. The nested sensitivity changes the assumed toxicity SD to `0.5`, `1` or `2` times
its data-generating value. True response surfaces, outcome SDs and the efficacy
model stay fixed; adaptive dose paths can differ.

Individual outcomes enter the GP fit separately. A hypothetical cohort of size
`r_k` uses cohort-mean noise variance `sigma^2 / r_k` in the lookahead update.

### Assumptions

The covariance allows different correlation lengths along the two dose axes.
It does not enforce monotonicity or a pharmacologic interaction form.
Efficacy–toxicity dependence and uncertainty in fitted hyperparameters are
omitted. Alternative kernels and likelihood families were not compared.

Allocation rules and response surfaces have a registration interface; the GP
model does not. If you change the model, use the same model and fitting settings
for every rule you compare.
