#tcga
#(A)
tcga<-read.csv2("tcga_coad_samples.csv",sep=",")
tmn<-read.csv2("tnm.csv",sep=",")

tmn$t_rank_corr<-as.numeric(tmn$t_rank_corr)
tmn$n_rank_corr<-as.numeric(tmn$n_rank_corr)
tmn$m_rank_corr<-as.numeric(tmn$m_rank_corr)

tmn_long <- melt(
  as.data.table(tmn),
  id.vars = "feature",
  measure.vars = patterns(
    corr = "_rank_corr$",
    pval = "_rank_p$"
  ),
  variable.name = "tmn"
)
tmn_long[tmn_long$tmn==1,"tmn"]<-"T"
tmn_long[tmn_long$tmn==2,"tmn"]<-"N"
tmn_long[tmn_long$tmn==3,"tmn"]<-"M"

tmn_long$corr<-as.numeric(tmn_long$corr)
tmn_long$pval<-as.numeric(tmn_long$pval)

tmn_long[, log10p := -log10(pval)]
tmn_long[, sig_group := fifelse(
  pval >= 0.1, "Not significant",
  fifelse(corr > 0, "Positive", "Negative")
)]
tmn_long[, sig_group := factor(sig_group,levels = c("Positive","Not significant","Negative"))]
tmn_long[, p_bin := cut(pval,breaks = c(-Inf, 0.01, 0.05, 0.1, Inf),
                        labels = c("≤0.01", "0.01–0.05", "0.05–0.1", ">0.1"),right = TRUE)]
tmn_long_filtered <- tmn_long[
  , .SD[any(sig_group != "Not significant")],
  by = feature]
fea<-tmn_long_filtered$feature %>% unique()

tmn_mat<-as.matrix(tmn[,c(2,4,6)])
rownames(tmn_mat)<-tmn$feature
tmn_mat<-tmn_mat[fea,]
p1<-pheatmap::pheatmap(tmn_mat,cluster_cols = T,cluster_rows = T,scale = "row")
feaClust <- (p1$tree_row)
fea_order<-feaClust$labels[feaClust$order]
f1<-fea_order[1:3]
f2<-fea_order[4:8]
fea_order[1:8]<-c(f2,f1)
tmn_long_filtered$feature<-factor(tmn_long_filtered$feature,levels = rev(fea_order))

p<-ggplot(tmn_long_filtered,aes(x = tmn,y = feature,size = p_bin,fill = sig_group)) +
  geom_point(shape = 21, color = "black", stroke = 0.4, alpha = 0.9) +
  scale_size_manual(values = c("≤0.01" = 4.5,"0.01–0.05" = 3,"0.05–0.1" = 2.5,">0.1"=2),name = "p-value")+
  scale_fill_manual(values = c("Positive" = "#E39A35","Not significant"= "white","Negative" = "#625D9E" )) +
  theme_bw() +
  theme(panel.grid = element_blank(),axis.text.x = element_text(angle = 0, vjust = 0.5),
        legend.title = element_blank())

#(B)
roc<-read.csv2("df_pred.csv",sep=",")
roc$AUC<-as.numeric(roc$AUC)
library(ggplot2)
roc$Outcome<-factor(roc$Outcome,levels = c("lymphatic_invasion","non_nodal_tumor_deposits","new_tumor_event_after_initial_treatment"))

p<-ggplot(roc, aes(x =Outcome , y = AUC, fill = Feature.Group)) +
  geom_boxplot(position = position_dodge(width = 0.6),outlier.size = 0,width = 0.4) +
  geom_jitter(aes(color = Feature.Group),position = position_jitterdodge(
    jitter.width = 0,dodge.width = 0.6),size = 1,alpha = 0.6) +ylim(0,1)+
  scale_fill_manual(values=c("#E39A35","#6A8CAF","#625D9E","#712820","#316D77"))+ 
  scale_color_manual(values=c("#E39A35","#6A8CAF","#625D9E","#712820","#316D77"))+ 
  theme_classic() 

#(C-D)
feature<-read.csv2("features.csv",sep=",")
rownames(feature)<-feature$sample
clinical<-read.csv2("clinical.csv",sep=",")[,-1]%>% as.data.frame()
common_samples<-intersect(feature$sample,clinical$sample)
clin_df <- clinical[clinical$sample %in%common_samples,]
data<-left_join(clin_df,feature)

results <- c()
p.list<-list()
for (colname in colnames(feature)[-1]) {
  x <- data[,colname]%>% as.numeric()
  if (all(is.na(x)) || sd(x, na.rm = TRUE) == 0) next

  q30 <- quantile(x, 0.3, na.rm = TRUE)
  low_mask  <- q30 > x 
  high_mask <- x > q30

  if (sum(high_mask) < 5 || sum(low_mask) < 5) next
  group <- rep(NA, length(x))
  group[low_mask]  <- "low"
  group[high_mask] <- "high"
  data$group<-group
  fit<- survfit(Surv(`OS.time`, OS) ~ group,data=data)
  #fit<- survfit(Surv(`DFI.time`, DFI) ~ group,data=data)
  p_value <-surv_pvalue(fit, data = data)$pval
  if(p_value<0.05){
    p<-ggsurvplot(fit,pval = T,conf.int = F,data=data ,
                  risk.table = F, # Add risk table
                  linetype = "solid", # Change line type by groups
                  #legend.labs = c("low","high"),#"Pattern1","Pattern2","Pattern3"
                  xlab = "Time in Days",
                  palette =c("#C09278","#4D4A6E" ))+
      labs(y="Disease-Free probability")#Disease-Free
    
    p.list[[colname]]<-p
  }
  results <- rbind(results,
                   data.frame(feature = colname,
                              p_value = p_value)
  )
}


#(E)

stroma<-read.csv2("stroma_barrier.csv",sep=",")[,-1]
colnames(stroma)[1]<-"sample"
stroma<-left_join(stroma,clinical[,c("sample","MSI")])
stroma<-stroma[!stroma$MSI %in% c("","Indeterminate"),]
stroma<-na.omit(stroma)
stroma$score<-as.numeric(stroma$score)
t.test(stroma[stroma$MSI=="MSI-H",]$score,stroma[stroma$MSI=="MSI-L",]$score,alternative = "less")
stroma$MSI<-factor(stroma$MSI,levels = rev(c("MSS","MSI-L","MSI-H")))

my_comparisons <- list(
  c("MSS", "MSI-L"),
  c("MSS", "MSI-H"),
  c("MSI-L", "MSI-H")
)
p<-ggplot(stroma,aes(x=MSI,y=score,fill=MSI))+
  geom_violin(trim = FALSE)+
  geom_boxplot(width=0.25,colour = "white")+
  scale_fill_manual(values=c("#E39A35","#316D77","#625D9E"))+
  stat_compare_means(
    comparisons = my_comparisons,method = "t.test", label = "p.signif",hide.ns = F ) +
  theme_classic()