# Import tools
library(lme4)
library(effects)
library(mgcv)
library(mgcViz)
library(ggplot2)
library(lmerTest)
library(car)
library(sandwich)

# Import data
data = read.delim("C:/Users/eliza/Desktop/Project/ASR_ICLDC/Results/merged_results.tsv")

# Data stuff:
# Convert rate to count data
data$cer_count = data$CER.scores * data$Utterance_Length

# Convert cols to factors
data$Target_Lang = as.factor(data$Target_Lang)
data$Strategy = as.factor(data$Strategy)
data$Utterance_ID = as.factor(data$Utterance_ID)
data$Utterance_Length = as.numeric(data$Utterance_Length)

# Okay now let's try a model; interaction between strat and target lang
opt = glmerControl(optimizer = "bobyqa")

m1 = glmer(cer_count ~ Strategy * Target_Lang + 
             offset(log(Utterance_Length)) + (1|Utterance_ID),
           data=data, family = poisson(link = log), control=opt)
save(m1, file="C:/Users/eliza/Desktop/Project/ASR_ICLDC/Results/m1.Rda", compress="xz")
#load("C:/Users/eliza/Desktop/Project/ASR_ICLDC/Results/m1.Rda")

summary(m1)
exp(fixef(m1))

# lets do some plots

# Compute effect
eff <- Effect(c("Strategy", "Target_Lang"), m1)
eff_df <- as.data.frame(eff)

# Recode labels
eff_df$Strategy <- factor(
  eff_df$Strategy,
  levels = c("fam", "geo", "phon", "rand"),
  labels = c("Genealogical", "Geographic", "Phonetic", "Random")
)

eff_df$Target_Lang <- factor(
  eff_df$Target_Lang,
  levels = c("hsb", "lg", "tt"),
  labels = c("Upper Sorbian", "Luganda", "Tatar")
)

# Colors for strategies
lang_cols <- c(
  "Upper Sorbian" = "#1b9e77",
  "Luganda"   = "#d95f02",
  "Tatar"     = "#7570b3"
)

ggplot(eff_df, aes(x = Strategy, y = fit, color = Target_Lang, group = Target_Lang)) +
  geom_point(size = 2.5) +
  geom_line(linewidth = 0.8) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15, linewidth = 0.5) +
  facet_wrap(~ Target_Lang) +
  scale_color_manual(values = lang_cols) +
  labs(
    title = "Predicted Error Counts by Strategy and Target Language",
    x = "Training Language Strategy",
    y = "Predicted Error Count",
    color = "Target Language"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 20, hjust = 1),
    strip.text = element_text(face = "bold"),
    legend.position = "none"
  )
