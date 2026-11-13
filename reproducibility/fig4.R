#xenium
#(A)
expr <- read.csv("D:\\work\\protein\\xenium\\pred_exp.csv", row.names = 1, check.names = FALSE)
coord <- read.csv("D:\\work\\protein\\xenium\\pos_test.csv", stringsAsFactors = FALSE)
celltpe <- read.csv("D:\\work\\protein\\xenium\\celltype.csv", stringsAsFactors = FALSE)
colnames(celltpe)[1]<-"barcode"

common_bc <- intersect(rownames(expr), coord$barcodes)

expr  <- expr[common_bc, ]
coord <- coord[match(common_bc, coord$barcodes), ]
protein_mat <- t(as.matrix(expr))  # 行=protein，列=barcode
seu <- CreateSeuratObject(
  counts = protein_mat,
  assay = "RNA",
  project = "Xenium_Protein"
)
coords <- data.frame(
  x = coord$col,
  y = coord$row,
  row.names = coord$barcodes
)
fov <- CreateFOV(
  coords = coords,
  type = "centroids"
)
seu[["Xenium"]] <- fov
DefaultFOV(seu) <- "Xenium"
seu$split <- coord$split

#(B)

xenium_rna <- readRDS("xenium_rna.rds")
xenium_rna$celltype<-factor(xenium_rna$celltype,levels = c("CD8 T","Macrophage","Epithelial","Undefined","Endothelial",
                                                           "Fibroblast","Naive CD4 T","DC","Plasma","Lymphatic Endothelial", "B"))

Idents(xenium_rna)<-xenium_rna$celltype
xenium_rna[["spatial"]] <- CreateDimReducObject(embeddings = coor, key = "SPATIAL_", assay = DefaultAssay(xenium_protein))
color<-c("#AA3C4F", '#8C549C',"#23452F",'#E39A35','#58A4C3','#6778AE','#D6E7A3',"#DCC1DD",
         '#F3B1A0','#76B6C3','#E0D4CA')
umap_tx = data.frame(xenium_rna@reductions$umap@cell.embeddings %>%  
                       as.data.frame())%>%cbind('Type' =xenium_rna$celltype)
centroids <- umap_tx %>%
  group_by(Type) %>%
  summarise(UMAP_1 = mean(UMAP_1), UMAP_2 = mean(UMAP_2))

p<-ggplot(umap_tx,aes(x=UMAP_1,y=UMAP_2,fill=Type,color=Type))+
  geom_point(size = 0.2,shape = 16, stroke = 0)+
  scale_color_manual(values=rev(color))+ #HES
  scale_fill_manual(values=rev(color))+
  geom_text(data = centroids,aes(label = Type),color = "black",size = 2.5
  ) +theme_classic()


#(C)
xenium_protein <- readRDS("xenium_protein.rds")
coor <- xenium_protein@reductions$spatial@cell.embeddings %>%  as.data.frame()
colnames(coor)<-c("UMAP_1",'UMAP_2')
rownames(coor)<-colnames(seu)
ymin <- min(coor$UMAP_2)
ymax <- max(coor$UMAP_2)
coor$UMAP_2 <- ( ymin + ymax) - coor$UMAP_2
coor<-as.matrix(coor)
xenium_protein[["umap"]] <- CreateDimReducObject(embeddings = coor, key = "UMAP_", assay = DefaultAssay(xenium_protein))
xenium_protein$barcode<-rownames(xenium_protein@meta.data)
xenium_protein@meta.data<-left_join(xenium_protein@meta.data,celltpe)
rownames(xenium_protein@meta.data)<-xenium_protein$barcode
xenium_protein$celltype<-factor(xenium_protein$celltype,levels = c("B","Plasma","Naive CD4 T","CD8 T","DC","Macrophage",
                                                                   "Fibroblast","Epithelial","Endothelial","Lymphatic Endothelial","Undefined"))
Idents(xenium_protein)<-xenium_protein$celltype
DimPlot(xenium_protein,pt.size = 0.1)+
  scale_color_manual(values=rev(color))+ #HES
  scale_fill_manual(values=rev(color)) +theme_classic()

#(E)

p2<-FeaturePlot(xenium_rna,features ="PTPRC",reduction = "spatial",pt.size =0.1)+
  labs(title = "PTPRC (RNA)",x="SPATIAL1",y="SPATIAL2")+
  scale_color_gradientn(colors =(c("lightgrey","#6778AE")))+
  theme_classic()
p3<-FeaturePlot(seu,features ="PTPRCRA",reduction = "umap",pt.size =0.1 )+
  labs(title =  "PTPRCRA (Predicted Protein)",x="SPATIAL1",y="SPATIAL2")+
  scale_color_gradient2(low="white", mid="lightgrey",high ="#6778AE",midpoint = 0)+
  theme_classic()
p4<-FeaturePlot(seu,features ="PTPRCRO",reduction = "umap",pt.size =0.1 )+
  labs(title =  "PTPRCRO (Predicted Protein)",x="SPATIAL1",y="SPATIAL2")+
  scale_color_gradient2(low="white", mid="lightgrey",high ="#6778AE",midpoint = 0)+
  theme_classic()

#(F)
cd163 <- readRDS("cd163.rds")
cd163<-cd163$data
lag3 <- readRDS("lag3.rds")
lag3<-lag3$data

colnames(cd163)[1:2]<-c("UMAP_1",'UMAP_2')
ymin <- min(cd163$UMAP_2)
ymax <- max(cd163$UMAP_2)
cd163$UMAP_2 <- ( ymin + ymax) - cd163$UMAP_2
cd163$light<-as.character(cd163$light)
cd163[cd163$light=="Gene1","light"]<-"MKI67"
cd163[cd163$light=="Gene2","light"]<-"CD163"
lag3$light<-as.character(lag3$light)
lag3[lag3$light=="Gene1","light"]<-"LAG3"
lag3[lag3$light=="Gene2","light"]<-"MKI67"
cd163[rownames(lag3[lag3$light=="LAG3",]),"light"]<-"LAG3"

p<-ggplot(cd163,aes(x=UMAP_1,y=UMAP_2,fill=light,color=light))+
  geom_point(data = subset(cd163,light=="Neither"),
             fill="#DCDCDC",color="#DCDCDC",size = 0.2,shape = 16, stroke = 0) +
  geom_point(data = subset(cd163,light!="Neither"),size = 0.3,shape = 16, stroke = 0.3)+
  scale_color_manual(values=c(MKI67='#AA3C4F',CD163='#E39A35',LAG3="#58A4C3"))+ 
  scale_fill_manual(values=c(MKI67='#AA3C4F',CD163='#E39A35',LAG3="#58A4C3"))+theme_classic()

#(G)
library(MASS)
data<-read.csv2("plot_pten.csv",sep=",")
data$target<-as.numeric(data$target)
data$pred<-as.numeric(data$pred)
data<-data[data$target!=min(data$target),][,2:3] #
get_density <- function(x, y, kde) {
  ix <- findInterval(x, kde$x)
  iy <- findInterval(y, kde$y)
  kde$z[cbind(ix, iy)]
}
kde <- kde2d(data$pred, data$target, n = 200, 
             h = c(max(sd(data$pred, na.rm = TRUE), 1e-6),
                   max(sd(data$target, na.rm = TRUE), 1e-6)))
data$density <- get_density(data$pred, data$target, kde)

#model <- lm(pred~target, data=data)
#summary(model)
res <- cor.test(data$pred,data$target,method = "spearman")#pearson
res
p<-ggplot(data, aes(x=pred, y=target)) +

  geom_point(size=0.25,aes(fill=density,color=density))+
  scale_color_gradient(low="white",mid="#AA3C4F", high ="#AA3C4F",midpoint=1)+
  scale_fill_gradient(low="white",mid="#AA3C4F",high ="#AA3C4F",midpoint=1)+ 
  geom_smooth(method = "lm", se=TRUE, color="black", formula = y ~ x)+theme_bw()

