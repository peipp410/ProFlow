#Bulk
#(A) all tissue
all_tissue_protein <- readRDS("all_tissue_protein.rds")
sample<-read.csv2("sample.csv",header = T,sep=",")
colnames(sample)[1]<-"orig.ident"
all_tissue_protein@meta.data<-left_join(all_tissue_protein@meta.data,sample[,c(1,3)])
rownames(all_tissue_protein@meta.data)<-all_tissue_protein$orig.ident
colnames(all_tissue_protein@meta.data)[11]<-"tissue"
Idents(all_tissue_protein)<-all_tissue_protein$tissue

umap_tx = data.frame(all_tissue_protein@reductions$umap@cell.embeddings %>%  
                       as.data.frame())%>%cbind('tissue' =all_tissue_protein$tissue)
centroids <- umap_tx %>%
  group_by(tissue) %>%
  summarise(UMAP_1 = mean(UMAP_1), UMAP_2 = mean(UMAP_2))

prop <- table(umap_tx$tissue)/dim(umap_tx)[1]
prop<-prop[order(prop,decreasing = T)] %>%data.frame()
colnames(prop)[1]<-"tissue"
prop$tissue<-factor(prop$tissue,levels = prop$tissue)
umap_tx$tissue<-factor(umap_tx$tissue,levels = prop$tissue)
p<-ggplot(prop, aes(x = tissue, y = Freq,fill=tissue)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values=my36colors)+
  theme_classic() + scale_y_continuous(expand = c(0,0))+
  labs(x = "Tissue",y = "Proportion",title = "Tissue proportion") +
  scale_y_continuous(labels = scales::percent_format()) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  
  ggplot(umap_tx,aes(x=UMAP_1,y=UMAP_2,fill=tissue,color=tissue))+
  geom_point(size = 0.8,shape = 16, stroke = 0)+
  scale_color_manual(values=my36colors)+ #HES
  scale_fill_manual(values=my36colors)+theme_classic()

#(B)
expr <- GetAssayData(all_tissue_protein)  # protein × sample
meta <- all_tissue_protein@meta.data
expr<-t(expr) %>%as.data.frame()
expr$orig.ident<-rownames(expr)
expr<-left_join(expr,meta[,c("orig.ident","tissue")],by="orig.ident")
expr<-expr[,-384]
avg_expr <- expr %>%
  dplyr::group_by(tissue) %>%
  dplyr::summarise(dplyr::across( where(is.numeric), mean, na.rm = TRUE),.groups = "drop") %>% data.frame()
rownames(avg_expr)<-avg_expr$tissue
avg_mat <- t(as.matrix(avg_expr[,-1]))
#cv <- apply(avg_mat, 1, function(x) sd(x) / mean(x))
sd_val <- apply(avg_mat, 1, sd) 
non_specific <- top_markers %>%
  dplyr::group_by(gene) %>%
  dplyr::summarise(n = n()) %>%data.frame()
#dplyr::filter(n >= n_tissue * 0.25)%>% 
sd_val<-data.frame(sd_val)
sd_val$gene<-rownames(sd_val)
colnames(sd_val)[1]<-"SD"
non_specific<-left_join(sd_val,non_specific)
non_specific[is.na(non_specific$n),"n"]<-0
avg_mat<-data.frame(avg_mat)
avg_mat$gene<-rownames(avg_mat)
non_specific<-left_join(non_specific,avg_mat)
non_specific<-non_specific %>%
  pivot_longer(
    cols = all_of(c(4:35)),
    names_to = "tissue",
    values_to = "avg"
  )

non_specific <- non_specific %>%
  dplyr::group_by(gene ) %>%
  dplyr::mutate( avg_mean = mean(avg, na.rm = TRUE))%>%data.frame()
non_specific<-non_specific[,-c(4,5)] %>% unique()

cancer_specific<-top_markers_uniq$gene%>%unique()
non_specific$group<-"common"
non_specific[non_specific$n==0,"group"]<-"stable"
non_specific[non_specific$gene %in% cancer_specific,"group"]<-"cancer-specific"
p<-ggplot(non_specific,aes(x=avg_mean,y=SD,color = group))+
  geom_point(data = subset(non_specific,group=="common"),fill="#91D0BE",color="#91D0BE",size=1) +
  #new_scale_color() + new_scale_fill()+
  geom_point(data = subset(non_specific,group=="stable"),fill="lightgrey",color="lightgrey",size=1) +
  geom_point(data = subset(non_specific,group=="cancer-specific"),fill="#B53E2B",color="#B53E2B",size=1.5)+
  theme_bw() 


#(C)
marker<-FindAllMarkers(all_tissue_protein,only.pos = T)
marker<-marker[marker$p_val_adj<0.05,]
top_markers<-marker[marker$avg_log2FC>0.25,]%>%group_by(cluster)
genes<-top_markers$gene[duplicated(top_markers$gene)]%>% unique()

top_markers_uniq<-top_markers[!top_markers$gene%in% genes,]
top_markers_uniq<-top_markers_uniq%>%group_by(cluster)%>%top_n(n=5,wt=(avg_log2FC))
tissue<-top_markers_uniq$cluster%>% unique()
mat<-get_scale_mat(all_tissue_protein[,all_tissue_protein$tissue %in% tissue],
                   features=top_markers_uniq$gene%>%unique(),col.min=-1.5,col.max=1.5 ,scale = TRUE)
mat<-data.frame(mat)
mat$tissue<-rownames(mat)
mat_long <- mat %>%
  pivot_longer(cols = 1:19, 
               names_to = "protein", 
               values_to = "expression") 
ggplot(mat_long,aes(x=tissue,y=expression,color=expression))+
  geom_point(aes(size=p_group, color=cor)) +
  scale_color_gradient2(low="#6F5981", mid="white", high="#CB5C4B", midpoint=0) +
  scale_size_manual(values=c(10,8,6,3)) +
  theme_minimal() +
  
  DotPlot(all_tissue_protein[,all_tissue_protein$tissue %in% tissue],features=top_markers_uniq$gene%>%unique(), 
          cluster.idents = F,col.min = -1.5,col.max = 1.5,dot.scale = 5,scale.by = "size")+
  scale_size_continuous(range = c(0.5,4))+
  scale_color_gradientn(colors = c("#316D77","#62B8D8","white","#F7C553","#E37321"))+coord_flip()

#(E-J)LUAD
luad<- readRDS("luad_plot.rds")
p<-DimPlot(luad,group.by = "cluster",pt.size = 0.8,label = T)+
  scale_color_manual(values=c('0'='#AA3C4F','1'='#E39A35','2'="#58A4C3"))+ 
  scale_fill_manual(values=c('0'='#AA3C4F','1'='#E39A35','2'="#58A4C3"))+theme_classic()


meta<-luad@meta.data[,c("subcluster","OS","OS.time")]
meta<-meta[meta$subcluster!="na",]%>% data.frame()
meta$OS<-as.numeric(meta$OS)
meta$OS.time<-as.numeric(meta$OS.time)
fit<- survfit(Surv(`OS.time`, OS) ~ subcluster, data = meta)  

p<-ggsurvplot(
  fit,
  data = meta,
  pval = TRUE,
  conf.int = F,
  risk.table = FALSE,
  palette = c("#C1B5D8","#8C549C" ),#58A4C3
  legend.title = "",
  title = paste0("Kaplan-Meier Survival Curve: Subcluster ", sub_c, " vs Others"),
  xlab = "Overall survival time",
  ylab = "Survival probability",
  ggtheme = theme_classic()
)



sub <- readRDS("luad_subcluster_plot.rds")
coor <- sub@reductions$umap@cell.embeddings %>%  as.data.frame()
rownames(coor)<-colnames(sub)
coor<-as.matrix(coor)
sub_rna[["umap"]] <- CreateDimReducObject(
  embeddings = coor,
  key = "UMAP_",
  assay = DefaultAssay(luad_rna)
)

DimPlot(sub,group.by = "sub_leiden",pt.size = 0.8,label = T)+
  scale_color_manual(values=c('1'='#8C549C','0'="#C1B5D8"))+ 
  scale_fill_manual(values=c('1'='#8C549C','0'="#C1B5D8"))+theme_classic()


gsea_hallmark<-list()
Idents(sub_rna)<-luad_rna$subcluster
marker<-FindAllMarkers(sub_rna,only.pos=FALSE,min.pct=0,logfc.threshold=0)
for (i in c(0,1)) {
  deg<-marker[marker$cluster==i,]
  deg<-deg[!grepl("^MT-|^RP[SL]",deg$gene),]
  GSEA_input <- deg$avg_log2FC
  names(GSEA_input) = deg$gene
  GSEA_input = sort(GSEA_input, decreasing = TRUE)
  GSEA_GO<-GSEA(GSEA_input,TERM2GENE = Hallmark_enrichR,pvalueCutoff = 1)
  gsea_hallmark[[paste0("c",i)]]<-GSEA_GO@result#GSEA_GO@result[,c(1:10)]
  
}
pathway <- lapply(gsea_hallmark, function(df) {
  df <- df[order(df$NES), ]
  df[1:25,"ID"]
})%>% unlist() %>% unique()
gsea_matrix<-matrix(0, nrow =length(pathway), ncol = 2)
colnames(gsea_matrix)<-c("c0","c1")
rownames(gsea_matrix)<-pathway

for (i in c("c0","c1")) {
  go<-data.frame(gsea_hallmark[[i]]$NES[match(pathway,gsea_hallmark[[i]]$Description)])
  colnames(go)<-i
  rownames(go)<-pathway
  rows <- intersect(rownames(go), pathway)
  gsea_matrix[rows, i] <- go[rows, ]
}
gsea_matrix[is.na(gsea_matrix)] <- 0
p_matrix<-matrix(0,  nrow =length(pathway), ncol =2)
colnames(p_matrix)<-c("c0","c1")
rownames(p_matrix)<-pathway

for (i in c("c0","c1")) {
  go<-data.frame(gsea_hallmark[[i]]$pvalue[match(pathway,gsea_hallmark[[i]]$Description)])
  colnames(go)<-i
  rownames(go)<-pathway
  rows <- intersect(rownames(go), pathway)
  p_matrix[rows, i] <- go[rows, ]
}
p_matrix[is.na(p_matrix)] <- "1"
p_matrix[p_matrix<0.05]<-"*"
p_matrix[p_matrix!="*"]<-""
min=round(range(gsea_matrix)[1])
max=ceiling(range(gsea_matrix)[2])
l=max-min
l0=abs(min-0)
bk <- c(seq(min,0,by=1.5),seq(0.1,max,by=1.5))
ph<-pheatmap::pheatmap(gsea_matrix,scale = "column", 
                       number_color = "white",fontsize_number=12,
                       show_colnames = T,show_rownames = T,cluster_rows=F,cluster_cols =F,
                       border_color = "white",
                       c("#5C8CAF", "#7FA2C2", "#C7CFE9","#E3E7F3","#DDA5C1", "#CC84A9","#944971","#3f1f30")    
)

p1<-FeaturePlot(sub,features = "CGAS")+theme_classic()+
  scale_color_gradient2(low="lightgrey", mid="lightgrey", high="#4EB3D3",midpoint = 0) 

p2<-FeaturePlot(sub,features = "STING1")+theme_classic()+
  scale_color_gradient2(low="lightgrey", mid="lightgrey", high="#CB181D",midpoint = 0.5)

p3<-FeaturePlot(sub_rna,features = "CGAS")+theme_classic()+
  scale_color_gradient2(low="lightgrey", mid="lightgrey", high="#4EB3D3",midpoint = 0) 

p4<-FeaturePlot(sub_rna,features = "STING1")+theme_classic()+
  scale_color_gradient2(low="lightgrey", mid="lightgrey", high="#CB181D",midpoint = 0.5)

#(K-M)LUSC
lusc_rna <- readRDS("lusc_rna.rds")
lusc <- readRDS("lusc1.rds")
lusc$orig.ident<-rownames(lusc@meta.data)
sample<-intersect(lusc$orig.ident,lusc_rna$name)
lusc_rna<-lusc_rna[,lusc_rna$name %in% sample]
geneset<-list(scc=c("SOX2", "TP63", "KRT6A", "CLCA2", "KRT5", "DSG3", "SFN","KLF5" ,"TP63","MKI67","RRM2"))

gsva_score = gsva(GetAssayData(lusc), geneset,  method="gsva",
                  kcdf="Poisson",ssgsea.norm = TRUE, verbose = TRUE)
gsva_long<-gsva_score  %>%t() %>%as.data.frame()
gsva_long$orig.ident<-rownames(gsva_long)
lusc@meta.data<-left_join(lusc@meta.data,gsva_long[,c("scc","orig.ident")])
rownames(lusc@meta.data)<-lusc$orig.ident

p<-DimPlot(lusc,pt.size = 1,group.by = "leiden")+labs(x="UMAP1",y="UMAP2")+
  scale_color_manual(values=(c( "#E37321",'#62B8D8','#F7C553','#316D77','#6778AE')))+
  scale_fill_manual(values=(c("#E37321",'#62B8D8','#F7C553','#316D77','#6778AE')))
p1<-FeaturePlot(lusc,features = c("RRM2"), order =T,pt.size = 1)+
  scale_color_gradientn(colors =(c("lightgrey","#FCEBEB" ,"#F08C8C","#AD1D41")))+
  labs(title =NULL)
p2<-FeaturePlot(lusc,features = c("scc"), order =T,pt.size = 1)+
  scale_color_gradientn(colors =(c("lightgrey","#CCC9E6" ,"#6778AE","#625D9E")))+
  labs(title =NULL)


os<-read.csv2("survival_LUSC_survival.txt",sep="\t",header = T)
os1<-read.csv2("TCGA-LUSC.survival.tsv",sep="\t",header = T)
os<-left_join(os,os1[,c("sample","X_PATIENT")])
os<-os[,-c(1:2,11)]
colnames(os)[9]<-"orig.ident"
lusc@meta.data<-left_join(lusc@meta.data,os)
rownames(lusc@meta.data)<-lusc$orig.ident
meta<-lusc@meta.data[,c("OS.time","OS","leiden")]
meta$group<-"others"
meta[meta$leiden %in% c("3"),"group"]<-"c1"
fit<- survfit(Surv(`OS.time`, OS) ~ group,data=meta)
p<-ggsurvplot(fit,pval = T,conf.int = T,data=meta,
              risk.table = FALSE, # Add risk table
              linetype = "solid", # Change line type by groups
              surv.median.line = "hv", # Specify median survival
              ggtheme = theme_bw(), # Change ggplot2 theme
              xlab = "Time in Months",
              palette = c("#625D9E","#F1BB72"))+ 
  labs(y="survival probability")#Disease-Free 


coor<-lusc@reductions$umap@cell.embeddings
colnames(coor)<-c('UMAP_1',"UMAP_2")
rownames(coor)<-colnames(lusc)
sample<-colnames(lusc_rna)
lusc_rna[["umap"]] <- CreateDimReducObject(
  embeddings = as.matrix(coor[sample,]),
  key = "UMAP_",assay = DefaultAssay(lusc_rna))
DimPlot(lusc_rna,reduction = "umap")

