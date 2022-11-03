library(ggplot2)
library(readr)
library(dplyr)

data.libraries <- read_csv("coster-so/libraries.csv")

# data.coster <- data.libraries %>% filter(Model == "COSTER")
# data.resico <- data.libraries %>% filter(Model == "RESICO-KNN")

ggplot(data.libraries, aes(Library)) + 
  geom_bar(data = subset(data.libraries), 
           aes(y = log10(Amount), fill = Type), stat = "identity", position = "dodge") +
  # geom_bar(data = subset(data.libraries) 
  #          aes(y = -Amount, fill = Type), stat = "identity", position = "dodge") + 
  geom_hline(yintercept = 0,colour = "grey90")


last_plot() + 
  geom_text(data = subset(df.m, variable == "count.up"), 
            aes(strain, value, group=condition, label=value),
            position = position_dodge(width=0.9), vjust = 1.5, size=4) +
  geom_text(data = subset(df.m, variable == "count.down"), 
            aes(strain, -value, group=condition, label=value),
            position = position_dodge(width=0.9), vjust = -.5, size=4) +
  coord_cartesian(ylim = c(-500, 500))
