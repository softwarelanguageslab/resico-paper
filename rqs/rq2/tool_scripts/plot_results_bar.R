library(ggplot2)
library(ggpattern)
library(readr)
library(dplyr)
library(scales)

data.intrinsic <- read_csv("results.csv")

# Plotting with ggplot2
ggplot(data = data.intrinsic, aes(x = Classifier, y = Score, fill=Metric)) +
  geom_col(colour="black", position = position_dodge2(padding=0.1)) +
  # geom_col_pattern(position = "dodge", aes(pattern = Approach, pattern_angle = Approach), pattern_spacing = 0.025) +
  geom_text(aes(y = Score, label = Score), vjust = -0.5, position=position_dodge(width=1),
            color = "black", size = 4.5, fontface = "bold") +
  # geom_hline(aes(yintercept = `Max-F1-RESICO`, linetype="RESICO"), colour = "black") +
  # geom_hline(aes(yintercept = `Max-F1-COSTER`, linetype="COSTER"), colour = "brown") +
  # scale_linetype_manual(name = "Best Scores", values = c(2, 2), 
  #                       guide = guide_legend(override.aes = list(color = c("black", "brown")))) +
  facet_wrap(~ `Top-K`, ncol = 1, nrow = 3) +
  scale_y_continuous(breaks = c(0, 25, 50, 75, 100),
                     labels = c("0%", "25%", "50%", "75%", "100%")) +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        legend.text = element_text(size=11),
        axis.text.x = element_text(size=13),
        axis.text.y = element_text(size=13),
        axis.title = element_text(size=15),
        strip.text = element_text(size = 13))
