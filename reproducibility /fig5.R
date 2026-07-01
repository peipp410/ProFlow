#HE

modify_vlnplot<- function(obj,
                          feature,
                          pt.size = 0,
                          plot.margin = unit(c(-0.75, 0, -0.75, 0), "cm"),
                          ...) {
  p<- VlnPlot(obj, features = feature, pt.size = pt.size, ... )  +
    xlab("") + ylab(feature) + ggtitle("") +
    theme(legend.position = "none",
          axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.x = element_blank(),
          axis.ticks.y = element_line(),
          axis.title.y = element_text(size = rel(1), angle = 0, vjust = 0.5),
          plot.margin = plot.margin )
  return(p)
}

## main function
StackedVlnPlot<- function(obj, features,
                          pt.size = 0,
                          plot.margin = unit(c(-0.75, 0, -0.75, 0), "cm"),
                          ...) {
  
  plot_list<- purrr::map(features, function(x) modify_vlnplot(obj = obj,feature = x, ...))
  plot_list[[length(plot_list)]]<- plot_list[[length(plot_list)]] +
    theme(axis.text.x=element_text(), axis.ticks.x = element_line())
  
  p<- patchwork::wrap_plots(plotlist = plot_list, ncol = 1)
  return(p)
}
#(B)
he <- readRDS("TCGA-5M-AAT5.rds")
he <- readRDS("TCGA-A6-2677.rds")
he$spot<-rownames(he@meta.data)
Idents(he)<-he$kmeans_cluster
DimPlot(he,reduction = "spatial")

coor<-he@reductions$spatial@cell.embeddings %>% data.frame()
colnames(coor)<-c("UMAP_1",'UMAP_2')
rownames(coor)<-colnames(he)
ymin <- min(coor$UMAP_2)
ymax <- max(coor$UMAP_2)
coor$UMAP_2 <- ( ymin + ymax) - coor$UMAP_2
coor<-as.matrix(coor)
he[["umap"]] <- CreateDimReducObject(embeddings = coor, key = "UMAP_", assay = DefaultAssay(he))


color=c('#F3B1A0','#E4C755','#DCC1DD', '#58A4C3','#6778AE','#23452F','#AA3C4F', '#CCE0F5','#E0D4CA',
        '#C5DEBA','#E39A35')
Idents(he)<-he$kmeans_cluster
p<-DimPlot(he,reduction = "umap",pt.size = 0.8)+
  scale_color_manual(values=rev(color))+ #HES
  scale_fill_manual(values=rev(color)) +theme_classic()

Idents(he)<-he$Tissue_Type
p<-DimPlot(he,reduction = "umap",pt.size = 0.8)+
  scale_color_manual(values=c('#b7c6b3','#d9cbb0','#8797a6','#f1ece6'))+ #HES
  scale_fill_manual(values=c('#b7c6b3','#d9cbb0','#8797a6','#f1ece6')) +theme_classic()

#(C)
p_scc<-plot_density(he, features = c("KRT19","CDH1"), reduction = "umap",joint = T )
p_scc<-p_scc[[3]]
p_scc_data<-p_scc$data
p<-ggplot(p_scc_data, aes(x=UMAP_1, y=UMAP_2 )) +
  geom_point(aes(color=feature),size = 0.8) +
  scale_color_gradient2(low="lightgrey", mid="white", high="#58A4C3",midpoint = 2.5e-15) +
  theme_minimal() +theme(
    axis.text = element_blank(), 
    axis.title  = element_blank(), 
    axis.ticks = element_blank(),
    panel.background = element_blank(),  
    panel.grid = element_blank(),        
    legend.text = element_blank(),legend.position = "none"
  ) 

#(D)
p_scc_data$spot<-rownames(p_scc_data)
p_scc_data<-left_join(p_scc_data,he@meta.data[,c("spot","kmeans_cluster")])

cell<-names(table(p_scc_data$kmeans_cluster))
clean_data<-c()
for (i in cell) {
  d<-p_scc_data[p_scc_data$kmeans_cluster==i,]
  Q1 <- quantile(d$feature, 0.25)
  Q3 <- quantile(d$feature, 0.75)
  IQR <- Q3 - Q1
  
  lower_bound <- Q1 + 0.4 * IQR
  upper_bound <- Q3 - 0.4 * IQR
  
  clean <- d[d$feature >= lower_bound & d$feature <= upper_bound,]
  clean$feature[clean$feature< -2] <- -2
  clean$feature[clean$feature>2] <- 2
  #clean$feature<- normalization(clean$feature)
  clean_data<-rbind(clean_data,clean)
}
plot_data <- clean_data %>%#density,clean_data
  group_by(kmeans_cluster) %>%
  summarise(
    mean_prop = mean(feature),
    sd_prop = sd(feature),
    .groups = "drop"
  )

plot_data<-plot_data[order(plot_data$mean_prop,decreasing = T),]
plot_data$kmeans_cluster<-factor(plot_data$kmeans_cluster,levels = plot_data$kmeans_cluster)
p<-ggplot(plot_data, aes(x = kmeans_cluster, y = mean_prop, fill = kmeans_cluster)) +
  geom_col(position = position_dodge(0.8), width = 0.6) +
  geom_text(
    aes(y =mean_prop+sd_prop, label = "—"),
    position = position_dodge(0.8), 
    vjust =0.3, size = 5
  ) +
  geom_text(
    aes(y =mean_prop-sd_prop, label = "—"),
    position = position_dodge(0.8), 
    vjust =0.3, size = 5
  )  +scale_y_continuous(expand = c(0,0))+
  scale_fill_manual(values = color) +labs(y="Mean Density")+
  theme_classic()


#(E)
color=c('#CCE0F5', '#6778AE','#23452F','#C5DEBA','#F3B1A0','#DCC1DD', '#58A4C3','#AA3C4F',
        '#E0D4CA', '#E4C755','#E39A35')
p1<-FeaturePlot(he,features = "PECAM1")+theme_classic()+
  scale_color_gradient2(low="lightgrey", mid="white", high="#6778AE",midpoint = 0) 

p2<-FeaturePlot(he,features = "CD3E")+theme_classic()+
  scale_color_gradient2(low="lightgrey", mid="white", high="#23452F",midpoint = 0) 

p3<-FeaturePlot(he,features = "MS4A1")+theme_classic()+
  scale_color_gradient2(low="lightgrey", mid="white", high="#AA3C4F",midpoint = 0) 

p4<-FeaturePlot(he,features = "CD68")+theme_classic()+
  scale_color_gradient2(low="lightgrey", mid="white", high="#0C2C84",midpoint = 0) 

p5<-FeaturePlot(he,features = "ACTA2")+theme_classic()+
  scale_color_gradient2(low="lightgrey", mid="white", high="#4EB3D3",midpoint = 0) 

p6<-FeaturePlot(he,features = "MKI67")+theme_classic()+
  scale_color_gradient2(low="lightgrey", mid="white", high="#CB181D",midpoint = 0) 

#(F)

he$cancer_region<-"others"#he$Tissue_Type %>% as.character()
he@meta.data[he$kmeans_cluster==0,"cancer_region"]<-"core"
he@meta.data[he$kmeans_cluster==4,"cancer_region"]<-"boundary"
he@meta.data[he$kmeans_cluster==9,"cancer_region"]<-"outer"
he@meta.data[he$kmeans_cluster==8,"cancer_region"]<-"necrosis border"
he@meta.data[he$kmeans_cluster==2,"cancer_region"]<-"necrosis"
he$cancer_region<-factor(he$cancer_region,levels = c("core","boundary","necrosis",
                                                     "necrosis border","outer","others"))
Idents(he)<-he$cancer_region

p<-DimPlot(he)+
  scale_color_manual(values=c('#f1ece6','#d9cbb0','#b7c6b3','#8797a6','#CCE0F5','#CCC9E6'))+ #HES
  scale_fill_manual(values=c('#f1ece6','#d9cbb0','#b7c6b3','#8797a6','#CCE0F5','#CCC9E6')) +theme_classic()

#(G)
he$kmeans_cluster<-factor(he$kmeans_cluster,levels = c(0,4,8,9,2,7,1,5,6,3))
Idents(he)<-he$kmeans_cluster
he$cancer_region<-factor(he$cancer_region,levels = c("core","boundary","necrosis",
                                                     "outer","necrosis border","others"))
Idents(he)<-he$cancer_region

StackedVlnPlot(he[,!he$cancer_region %in% c("necrosis","others")], features = c("PECAM1","CD3E","ACTA2","MKI67"),
               pt.size=1, cols=rev(color))

#(H)

ratio<-read.csv2("tissue_ratio.csv",sep=",")
ratio<-left_join(ratio,clinical)
ratio$Tumor<-as.numeric(ratio$Tumor)
ratio$Stroma<-as.numeric(ratio$Stroma)
ratio$Immune<-as.numeric(ratio$Immune)
ratio$Background.Necrosis<-as.numeric(ratio$Background.Necrosis)

# M 
m_stage_mapping <- c(
  "M0" = 0,
  "M1" = 1,
  "MX" = NA
)

# N 
n_stage_mapping <- c(
  "N0" = 0,
  "N1" = 1,
  "N2" = 2
)

# T 
t_stage_mapping <- c(
  "T1" = 0,
  "T2" = 1,
  "T3" = 2,
  "T4" = 3
)

ratio <- ratio %>%
  mutate(
    pathologic_M = m_stage_mapping[pathologic_M],
    pathologic_N = n_stage_mapping[pathologic_N],
    pathologic_T = t_stage_mapping[pathologic_T]
  )

rownames(ratio)<-ratio$sample
ratio<-ratio[,-c(1,8)]
ratio$OS<-as.numeric(ratio$OS)
ratio$OS.time<-as.numeric(ratio$OS.time)
ratio$DFI<-as.numeric(ratio$DFI)
ratio$DFI.time<-as.numeric(ratio$DFI.time)
ratio[(ratio$MSI==""|is.na(ratio$MSI)),"MSI"]<-"NA"

ratio<-ratio[order(ratio$Tumor,decreasing = F),]

col_fun_OS <- colorRamp2(c(min(ratio$OS.time%>% na.omit()), mean(ratio$OS.time%>% na.omit()), max(ratio$OS.time%>% na.omit())),
                         c("#6A8CAF", "white", "#E59866")) 

col_fun_DFI <- colorRamp2(c(min(ratio$DFI.time%>% na.omit()), mean(ratio$DFI.time%>% na.omit()), max(ratio$DFI.time%>% na.omit())),
                          c("#80B1D3", "white", "#712820")) 
library(RColorBrewer)
mat <- as.matrix(t(ratio["OS.time"]))

col_OS <- c("0"="#8FB9A8", "1"="#DD8452")
col_DFI <- c("0"="#8FB9A8", "1"="#DD8452")
col_N       <- c("0"="#B2DF8A", "1"="#E1EEDB",'2'="#8DD3C7")
col_M        <- c("0"="#FEE6A3", "1"="#F0CA96")
col_T    <-c("0"="#5A8CA5", "1"="#C39BB0",'2'="#FBB4AE",'3'="#BEBADA")
col_MSI <- c( "Indeterminate" = "#4C72B0","MSI-L" = "#316D77","MSS" = "#F4C22D","MSI-H" = "#E39A35",'NA'="lightgrey")


top_anno <- HeatmapAnnotation(
  OS.Time     = ratio$OS.time,
  OS.Status   = ratio$OS,
  pathologic.T = ratio$pathologic_T,
  pathologic.N = ratio$pathologic_N,
  pathologic.M = ratio$pathologic_M,
  DFI      = ratio$DFI,
  DFI.Time      = ratio$DFI.time,
  MSI     = ratio$MSI,
  
  col = list(
    OS.Time     = col_fun_OS,
    OS.Status   = col_OS,
    pathologic.T = col_T,
    pathologic.N = col_N,
    pathologic.M = col_M,
    DFI      = col_DFI,
    DFI.Time      = col_fun_DFI,
    MSI     = col_MSI
  ),
  border = TRUE,
  gp = gpar(col = "white",lwd = 0.8), 
  annotation_height = unit(6, "mm")
)

p<-Heatmap(mat,
           name = "OS.Time",
           col = col_fun_OS,
           cluster_rows = F,
           cluster_columns = F,
           row_names_side = "left",
           show_column_names = TRUE,
           top_annotation = top_anno
           #heatmap_legend_param = list(title = "OS.Time")
)

ratio_long<-ratio[,1:5]%>% pivot_longer(cols =2:5 , names_to = "region", values_to = "prop") 
ratio_long$sample<-factor(ratio_long$sample,levels = ratio$sample)
p<-ggplot(ratio_long,aes(x=sample,y=prop,fill=region))+
  geom_bar( position = "stack", stat = "identity")+
  scale_fill_manual(values=c("#EFE6C8","#EBCB92","#CCC9E6","#C1B5D8"))+ theme_classic()+
  scale_y_continuous(expand = c(0,0))+
  labs(x = "Sample",y = "Proportion",title = "Tissue proportion") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
