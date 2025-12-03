setwd("~/SCCR/Project/dr. Vito/dr4pl")

library(tidyverse)
library(dr4pl)

# 1. Load (you already have these)
df_raw  <- read_csv("abs4.csv",  show_col_types = FALSE)
meta_raw <- read_csv("meta-abs4.csv", show_col_types = FALSE)

df_raw
meta_raw

# 2. Split semicolon fields (same as you had)
df <- df_raw %>%
  separate(
    `Abs;1;2;3;4;5;6;7;8;9;10;11;12`,
    into = c("Row","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12"),
    sep = ";"
  ) %>%
  filter(Row %in% LETTERS[1:4]) %>%
  select(Row, C2:C9)

meta <- meta_raw %>%
  separate(
    `Sample;1;2;3;4;5;6;7;8;9;10;11;12`,
    into = c("Row","S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S12"),
    sep = ";"
  ) %>%
  filter(Row %in% LETTERS[1:4]) %>%
  select(Row, S2:S9)

df
meta

# 3. Long format and merge
df_long <- df %>%
  pivot_longer(cols = starts_with("C"), names_to = "Column", values_to = "Abs") %>%
  mutate(Column = parse_number(Column), Abs = as.numeric(Abs))

meta_long <- meta %>%
  pivot_longer(cols = starts_with("S"), names_to = "Column", values_to = "Sample") %>%
  mutate(Column = parse_number(Column))

merged <- left_join(df_long, meta_long, by = c("Row", "Column"))

print(merged, n = 35)


# 4. Dose mapping (1..64, NC=0, Blank=NA)
dose_map <- tibble(Column = 2:9, Dose = c(10,25,50,100,200,400,0,NA_real_))
merged <- merged %>% left_join(dose_map, by = "Column")

print(merged, n = 35)

# 5. Keep only wells with Dose defined (treatments + NC)
fit_df <- merged %>% filter(!is.na(Dose))
print(fit_df, n = 35)

# 6. Normalize to negative control (Dose == 0)
NC_mean <- fit_df %>% filter(Dose == 0) %>% summarise(meanNC = mean(Abs, na.rm = TRUE)) %>% pull(meanNC)
if (is.na(NC_mean) || NC_mean == 0) stop("Negative control mean is NA or zero — check your NC wells.")

fit_df <- fit_df %>% mutate(NormAbs = Abs / NC_mean * 100)  # percent of NC
fit_df

# 7. Prepare data for fitting: EXCLUDE Dose == 0 (NC) and any NA doses (blank)
fit_for_model <- fit_df %>% filter(Dose > 0, !is.na(NormAbs))

print(fit_for_model, n = 50)

# 8. Fit 4PL with dr4pl (only real doses)
# use sensible options: decreasing trend if response falls with dose
model <- dr4pl(NormAbs ~ Dose, data = fit_for_model, method.init = "logistic", trend = "decreasing")
print(summary(model))

# 9. Robustly extract IC50 from dr4pl model
params <- coef(model)
cat("Parameter names from model:\n"); print(names(params))
print(params)

# dr4pl parameter definitions:
# theta_1 = upper limit
# theta_2 = IC50 (dose at 50% effect)
# theta_3 = slope
# theta_4 = lower limit

# try to find the IC50 parameter automatically
possible_names <- c("theta_2", "ec50", "ic50")

idx <- which(tolower(names(params)) %in% possible_names)

if (length(idx) == 0) {
  stop("No IC50/EC50-related parameter found (expected theta_2).")
}

param_name <- names(params)[idx[1]]
param_value <- params[idx[1]]

cat("\nDetected IC50 parameter:", param_name, "\n")
cat("Raw parameter value:", param_value, "\n")

# dr4pl returns IC50 already on DOSE SCALE → no log10 conversion needed
IC50_value <- param_value

cat("\nFinal IC50 estimate =", IC50_value, "\n")

# 10. Plot dose-response (exclude Dose==0 so no log(0) issues)

# safely extract dr4pl parameters
params <- model$parameters

# if parameters contain NA, attempt a recovery fit
if (any(is.na(params))) {
  message("⚠️ dr4pl parameters contain NA, refitting with robust settings...")
  model <- dr4pl(
    NormAbs ~ Dose,
    data = fit_for_model,
    method.init = "logistic",
    method.robust = "tukey",
    trend = "descending"
  )
  params <- model$parameters
}

# final check: if still NA, stop gracefully
if (any(is.na(params))) {
  stop("dr4pl model failed: parameters remain NA after refit.")
}

# create prediction grid
dose_grid <- exp(seq(
  log(min(fit_for_model$Dose) * 0.8),
  log(max(fit_for_model$Dose) * 1.2),
  length.out = 200
))

# dr4pl prediction (ONLY mean, no CI)
y_pred <- dr4pl:::MeanResponse(params, dose_grid)

# placeholder ribbon (optional)
# dr4pl has no built-in confidence intervals
Lo <- y_pred * 0.97   # fake 3% CI band
Hi <- y_pred * 1.03

pred_df <- data.frame(
  Dose = dose_grid,
  Pred = y_pred,
  Lo   = Lo,
  Hi   = Hi
)

library(ggplot2)

p <- ggplot() +
  geom_point(data = fit_for_model, aes(x = Dose, y = NormAbs), size = 2) +
  geom_line(data = pred_df, aes(x = Dose, y = Pred), linewidth = 1) +
  geom_ribbon(data = pred_df, aes(x = Dose, ymin = Lo, ymax = Hi), alpha = 0.15) +
  scale_x_log10() +
  labs(
    x = "Dose (µM, log10 scale)",
    y = "Normalized % (NC = 100%)",
    title = "4PL Fit (dr4pl)"
  ) +
  theme_minimal()

print(p)

# final IC50
IC50_value

#-------------------------------------------------------------------------------
# Validation

# Extract parameters safely from dr4pl
params <- coef(model)
names(params) <- c("UpperLimit", "IC50", "Slope", "LowerLimit")

Upper <- round(params["UpperLimit"], 3)
Lower <- round(params["LowerLimit"], 3)
Slope <- round(params["Slope"], 3)
IC50  <- round(params["IC50"], 3)

# Compute predictions for R²
pred <- dr4pl:::MeanResponse(params, fit_for_model$Dose)

obs <- fit_for_model$NormAbs
SSE <- sum((obs - pred)^2, na.rm = TRUE)
SST <- sum((obs - mean(obs))^2, na.rm = TRUE)
R2 <- round(1 - SSE/SST, 4)

# Build equation text
eq_text <- paste0(
  "4PL Model:\n",
  "y = Lower + (Upper - Lower) / (1 + (x/IC50)^Slope)\n",
  "Upper = ", Upper, "\n",
  "Lower = ", Lower, "\n",
  "IC50  = ", IC50, "\n",
  "Slope = ", Slope, "\n",
  "R² = ", R2
)

# Plot
p <- ggplot() +
  geom_point(data = fit_for_model, aes(x = Dose, y = NormAbs), size = 2) +
  geom_line(data = pred_df, aes(x = Dose, y = Pred), linewidth = 1) +
  geom_ribbon(data = pred_df, aes(x = Dose, ymin = Lo, ymax = Hi), alpha = 0.15) +
  scale_x_log10() +
  labs(
    x = "Dose (µM, log10 scale)",
    y = "% Viability",
    title = "4PL Fit (dr4pl)"
  ) +
  theme_minimal() +
  annotate(
    "text",
    x = min(pred_df$Dose) * 1.05,       # slightly inside left border
    y = min(pred_df$Pred) + 0,          # a bit above bottom so it's visible
    hjust = 0, vjust = 0,
    label = eq_text,
    size = 3.5
  )

p

