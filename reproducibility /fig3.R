#cptac
get_scale_mat<-function(object,features,col.min,col.max ,scale = c(TRUE,FALSE)){
  cells <- unlist(x = CellsByIdentities(object = object))
  data.features <- FetchData(object = object, vars = (features), 
                             cells = cells)
  data.features$id <-Idents(object = object)[cells, drop = TRUE]
  
  if (!is.factor(x = data.features$id)) {
    data.features$id <- factor(x = data.features$id)
  }
  id.levels <- levels(x = data.features$id)
  data.features$id <- as.vector(x = data.features$id)
  
  data.plot <- lapply(X = unique(x = data.features$id), FUN = function(ident) {
    data.use <- data.features[data.features$id == ident, 
                              1:(ncol(x = data.features) - 1), drop = FALSE]
    avg.exp <- apply(X = data.use, MARGIN = 2, FUN = function(x) {
      return(mean(x = expm1(x = x)))
    })
    pct.exp <- apply(X = data.use, MARGIN = 2, FUN = PercentAbove, 
                     threshold = 0)
    return(list(avg.exp = avg.exp, pct.exp = pct.exp))
  })
  names(x = data.plot) <- unique(x = data.features$id)
  
  data.plot <- lapply(X = names(x = data.plot), FUN = function(x) {
    data.use <- as.data.frame(x = data.plot[[x]])
    data.use$features.plot <- rownames(x = data.use)
    data.use$id <- x
    return(data.use)
  })
  data.plot <- do.call(what = "rbind", args = data.plot)
  if (!is.null(x = id.levels)) {
    data.plot$id <- factor(x = data.plot$id, levels = id.levels)
  }
  ngroup <- length(x = levels(x = data.plot$id))
  
  avg.exp.scaled <- sapply(X = unique(x = data.plot$features.plot), 
                           FUN = function(x) {
                             data.use <- data.plot[data.plot$features.plot == 
                                                     x, "avg.exp"]
                             if (scale) {
                               data.use <- scale(x = data.use)
                               data.use <- MinMax(data = data.use, min = col.min, 
                                                  max = col.max)
                             }
                             else {
                               data.use <- log1p(x = data.use)
                             }
                             return((data.use))
                           })
  rownames(avg.exp.scaled)<-id.levels
  return(avg.exp.scaled)
}

#(A-C)
metric_df <- read.csv("metrics.csv", row.names=1)
m_median <- median(metric_df$mean_corr, na.rm = TRUE)
s_median <- median(metric_df$std_corr, na.rm = TRUE)

groupA <- metric_df %>%
  filter(mean_corr < m_median, std_corr < s_median)
groupB <- metric_df %>%
  filter(mean_corr < m_median, std_corr > s_median)


metric_df <- metric_df %>%
  mutate(region = case_when(
    mean_corr < m_median & std_corr < s_median ~ "low mean and low std",
    mean_corr < m_median & std_corr >= s_median ~ "low mean and high std",
    mean_corr >= m_median ~ "high mean",
    TRUE ~ "Others"
  ))
model <- lm(std_corr~mean_corr, data=metric_df)
summary(model)
res <- cor.test(metric_df$mean_corr,metric_df$std_corr,method = "pearson")
res
p<-ggplot(metric_df, aes(x = mean_corr, y = std_corr, color = cv)) +
  geom_point(alpha = 0.7,size = 1.2,shape = 16, stroke = 0) +
  scale_color_gradientn(colors = (c("white",'#CCC9E6',"#C3A6C4","#997FA2" ,"#6F5981","#4E3F66")),)+
  geom_smooth(method = "lm", se=TRUE, color="black", formula = y ~ x)+
  annotate("text",label="pearson=-0.72~p<2.2e-16",
           parse=T,x=0.5,y=0.07,color="black",size=4)+
  labs(title = "Predictability vs Stability",x="Mean Spearman",y="Sd of Spearman")+theme_classic()

p<-ggplot(metric_df, aes(x = mean_corr, y = std_corr, color = region)) +
  geom_point(alpha = 0.7,size = 1.2,shape = 16, stroke = 0) +
  geom_segment(aes(x = m_median, xend = m_median,
                   y = min(std_corr), yend = max(std_corr)),
               linetype = "dashed", color = "black") +
  geom_segment(aes(x = min(mean_corr), xend = m_median,
                   y = s_median, yend = s_median),
               linetype = "dashed", color = "black") +
  scale_color_manual(values = c(
    "low mean and low std" = "#F3B1A0",
    "low mean and high std" = "#C1E6F3",
    "high mean" = "#C5DEBA",
    "others" = "grey80"
  )) +
  labs(x = "Mean Spearman", y = "Std of Spearman") +
  theme_minimal() +
  theme(
    legend.position = "none",   
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()  
  )

group1<-metric_df[metric_df$region=="high mean",]%>% top_n(n=9,wt=(mean_corr)) 
group1<-group1[order(group1$mean_corr,decreasing = T),]

group2<-metric_df[metric_df$region=="low mean and high std",]%>% top_n(n=9,wt=(std_corr)) 
group2<-group2[order(group2$std_corr,decreasing = T),]

group3<-metric_df[metric_df$region=="low mean and low std",]%>% top_n(n=-9,wt=(std_corr*mean_corr))

ggplot(metric_df, aes(x = mean_corr, y = std_corr, color = group)) +
  geom_point(alpha = 0.7,size = 1.2,shape = 16, stroke = 0) +
  scale_color_manual(values = c("group3" = "#F3B1A0","group2" = "#C1E6F3",
                                "group1" = "#C5DEBA","others" = "lightgrey")) +
  labs(title = "Predictability vs Stability",x="Mean Spearman",y="Sd of Spearman")+theme_classic()

DotPlot(cptac_exp,features = c(rownames(group1),rownames(group2),rownames(group3)),
        cluster.idents = F,col.min = -1.5,col.max = 1.5,dot.scale = 5,scale.by = "size")+
  scale_size_continuous(range = c(0.5,4))+
  scale_color_gradientn(colors = c("#316D77","#62B8D8","white","#F7C553","#E37321"))+coord_flip()

mat<-get_scale_mat(cptac_exp,features = c(rownames(group1),rownames(group2),rownames(group3)),
                   col.min=-1.5,col.max=1.5 ,scale = TRUE)

annotation <- data.frame(group =factor(c(rep("group1",dim(group1)[1]),rep("group2",dim(group2)[1]),
                                         rep("group3",dim(group3)[1])),levels =c("group1","group2","group3")))
rownames(annotation) <- c(rownames(group1),rownames(group2),rownames(group3))
annotation$protein<-rownames(annotation)
module_levels <-paste0("group",c(1:3))
module_cols<-structure(c("#C5DEBA","#C1E6F3","#F3B1A0"), names = module_levels)

ht<-Heatmap(scale(t(mat))%>% t() ,
            name = "proteins",
            cluster_rows = TRUE,      
            cluster_columns = T,
            #row_split = annotation$group,   
            column_split = annotation$group, 
            show_row_names = T,
            row_names_side = "left",   
            row_names_gp = gpar(fontsize = 8, col = "black"), 
            #row_labels = row_labels,    
            show_column_names = FALSE,
            col=c("#5394A7","#76ADBD","#9AC7D4","#CCE3E9","#E8D8BF","#AA6360","#974343"  ,"#852426"),
            row_gap = unit(0.1, "mm"),  
            column_gap = unit(0.1, "mm"),
            top_annotation = HeatmapAnnotation(group = annotation$group,
                                               col = list(group =module_cols))
)
draw(ht)

#(D)
gsea_hallmark<-list()
gsea_kegg<-list()
gsea_go<-list()
for (i in c("high mean","low mean and high std","low mean and low std")) {
  gene<-metric_df[metric_df$region==i,]$gene
  enrich_result <- enricher(gene =gene,TERM2GENE = Hallmark_enrichR,pvalueCutoff = 1)
  t<-enrich_result@result[enrich_result@result$p.adjust<0.05,]#pvalue
  gsea_hallmark[[i]]<-t
  
  enrich_result <- enricher(gene =gene,TERM2GENE = kegg_pathway,pvalueCutoff = 1)
  t<-enrich_result@result[enrich_result@result$p.adjust<0.05,]
  gsea_kegg[[i]]<-t
  
  enrich_result <- enricher(gene =gene,TERM2GENE = go_pathway,pvalueCutoff = 1)
  t<-enrich_result@result[enrich_result@result$p.adjust<0.05,]
  gsea_go[[i]]<-t
}

p.list<-list()
for (i in c("high mean","low mean and high std","low mean and low std")) {
  enrich<-gsea_hallmark[[i]]
  enrich$ID<-gsub("\\s*\\([^)]*\\)", "", enrich$ID)
  enrich <- enrich %>%
    filter(p.adjust < 0.05) %>%
    #mutate(Regulation = ifelse(NES > 0, "Up", "Down"))%>%
    mutate(logP = -log10(pvalue))
  p<-ggplot() +
    geom_segment(data = enrich,aes(x = 0, xend = logP, y = reorder(ID, logP), yend = ID,
                                   color = logP),linewidth = 0.8) +
    geom_point(data = enrich,aes(x = logP, y = ID, color = logP, size = Count)) +
    scale_color_gradient(low = "pink",high = "#AD1D41",name = "-log10(pvalue)") +
    scale_size_continuous(range = c(2, 5)) +
    ylab("") +xlab("logP") +labs(title = i,size = "Count") +#Set Size
    theme_bw() +
    theme(
      axis.text = element_text(colour = "black", size = 8),
      plot.title = element_text(size = 10, color = "black", face = "bold", vjust = 2, hjust = 0),
      axis.title = element_text(size = 8, color = "black", face = "plain", vjust = 1, hjust = 0.5),
      plot.margin = unit(c(0.25, 0.25, 0.25, 0.25), units = "cm"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      #panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank(),
    )
  p.list[[i]]<-p
}
cowplot::plot_grid(plotlist = p.list,ncol=3)

#(E-F)
cptac_exp<-readRDS("all_tissue_protein.rds")
sample<-read.csv2("sample.csv",sep=",")
colnames(sample)<-c("orig.ident","tissue")
cptac_exp@meta.data<-left_join(cptac_exp@meta.data,sample)
rownames(cptac_exp@meta.data)<-cptac_exp$orig.ident
Idents(cptac_exp)<-cptac_exp$tissue
DimPlot(cptac_exp)
expr <- GetAssayData(cptac_exp)

sample_detected <- colSums(!is.na(expr))
range(sample_detected)
protein_detected_ratio <- rowSums(!is.na(expr))
range(protein_detected_ratio)

protein_median <- apply(expr, 1, max, na.rm = TRUE)
range(protein_median)
remove_outlier_iqr <- function(x, k = 1.5) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower <- q1 - k * iqr
  upper <- q3 + k * iqr
  x[x > upper] <- 1.5
  x
}
expr_clean <- t(apply(expr, 1, remove_outlier_iqr))
cptac_exp@assays$RNA@counts<-expr_clean
cptac_exp <- NormalizeData(cptac_exp)
cptac_exp <- FindVariableFeatures(cptac_exp)
cptac_exp <- ScaleData(cptac_exp)
cptac_exp <- RunPCA(cptac_exp, features = VariableFeatures(object = cptac_exp),npcs = 50)
cptac_exp <- RunUMAP(cptac_exp, reduction = "pca", dims = 1:20)
DimPlot(cptac_exp)
scptac<-list()
for (i in c('brca','gbm','luad','lusc','ucec')) {
  path<-paste0(".\\",i,"_protein.rds")
  data<-readRDS(path)
  data$tissue<-i
  data$orig.ident<-rownames(data@meta.data)
  Idents(data)<-data$leiden
  cptac[[i]]<-data
}


sd_max<-metric_df[metric_df$std_corr==max(metric_df$std_corr),]%>% rownames()
spearman_max<-metric_df[metric_df$mean_corr==max(metric_df$mean_corr),]%>% rownames()

p<-DimPlot(cptac_exp,pt.size = 1)+labs(x="UMAP1",y="UMAP2")+
  scale_color_manual(values=c(luad="#E95C59",lusc="#316D77",gbm="#625D9E",brca="#712820",ucec="#E39A35"))+
  scale_fill_manual(values=c(luad="#E95C59",lusc="#316D77",gbm="#625D9E",brca="#712820",ucec="#E39A35"))