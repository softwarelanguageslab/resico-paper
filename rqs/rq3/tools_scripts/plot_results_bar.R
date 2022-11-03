library(ggplot2)
library(readr)
library(dplyr)
library(scales)
library(cowplot)

data.extrinsic <- read_csv("results.csv")

data.coster.so <- data.extrinsic %>% filter(Dataset == "COSTER-SO")
data.stattype.so <- data.extrinsic %>% filter(Dataset == "StatType-SO")
data.resico.so <- data.extrinsic %>% filter(Dataset == "RESICO-SO")

# Plotting for COSTER-SO
plot.coster <- ggplot(data = data.coster.so, aes(x = Classifier, y = Score, fill=Metric)) +
  geom_col(colour="black", position = position_dodge2(padding=0.1), width = 0.9) + 
  geom_text(aes(y = Score, label = Score), vjust = -0.5, position=position_dodge(width=0.9),
            color = "black", size = 4.5, fontface = "bold") +
  # geom_hline(aes(yintercept = `Max-F1-RESICO`, linetype="RESICO"), colour = "black") +
  # geom_hline(aes(yintercept = `Max-F1-COSTER`, linetype="COSTER"), colour = "brown") +
  # scale_linetype_manual(name = "Best Scores", values = c(2, 2), 
                        # guide = guide_legend(override.aes = list(color = c("black", "brown")))) +
  facet_wrap(~ `Top-K`, ncol = 3, nrow = 1) +
  scale_y_continuous(breaks = c(0, 25, 50, 75, 100),
                     labels = c("0%", "25%", "50%", "75%", "100%")) +
  xlab("") +
  ylab("") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        legend.position = "none",
        axis.text.x = element_text(size=13),
        axis.text.y = element_text(size=13),
        axis.title = element_text(size=15),
        strip.text = element_text(size = 13))


# Plotting for StatType-SO
plot.stattype <- ggplot(data = data.stattype.so, aes(x = Classifier, y = Score, fill=Metric)) +
  geom_col(colour="black", position = position_dodge2(padding=0.1), width = 0.9) + 
  geom_text(aes(y = Score, label = Score), vjust = -0.5, position=position_dodge(width=0.9),
            color = "black", size = 4.5, fontface = "bold") +
  # geom_hline(aes(yintercept = `Max-F1-RESICO`, linetype="RESICO"), colour = "black") +
  # geom_hline(aes(yintercept = `Max-F1-COSTER`, linetype="COSTER"), colour = "brown") +
  # scale_linetype_manual(name = "Best Scores", values = c(2, 2), 
  #                       guide = guide_legend(override.aes = list(color = c("black", "brown")))) +
  facet_wrap(~ `Top-K`, ncol = 3, nrow = 1) +
  scale_y_continuous(breaks = c(0, 25, 50, 75, 100),
                     labels = c("0%", "25%", "50%", "75%", "100%")) +
  xlab("") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        legend.position = "none",
        axis.text.x = element_text(size=13),
        axis.text.y = element_text(size=13),
        axis.title = element_text(size=15),
        strip.text = element_text(size = 13))


# Plotting for RESICO-SO
plot.resico <- ggplot(data = data.resico.so, aes(x = Classifier, y = Score, fill=Metric)) +
  geom_col(colour="black", position = position_dodge2(padding=0.1), width = 0.9) + 
  geom_text(aes(y = Score, label = Score), vjust = -0.5, position=position_dodge(width=0.9),
            color = "black", size = 4.5, fontface = "bold") +
  # geom_hline(aes(yintercept = `Max-F1-RESICO`, linetype="RESICO"), colour = "black") +
  # geom_hline(aes(yintercept = `Max-F1-COSTER`, linetype="COSTER"), colour = "brown") +
  # scale_linetype_manual(name = "Best Scores", values = c(2, 2), 
  #                       guide = guide_legend(override.aes = list(color = c("black", "brown")))) +
  facet_wrap(~ `Top-K`, ncol = 3, nrow = 1) +
  scale_y_continuous(breaks = c(0, 25, 50, 75, 100),
                     labels = c("0%", "25%", "50%", "75%", "100%")) +
  ylab("") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        legend.position = "bottom",
        legend.text = element_text(size=11),
        axis.text.x = element_text(size=13),
        axis.text.y = element_text(size=13),
        axis.title = element_text(size=15),
        strip.text = element_text(size = 13))

plot_grid(plot.coster, plot.stattype, plot.resico, labels = c("COSTER-SO", "StatType-SO", "RESICO-SO"), ncol = 1)
