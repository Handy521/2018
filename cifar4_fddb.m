clear;
%minimum size of face
minsize=40;
caffe_model_path='./model';
gpu_id=0;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);
%three steps's threshold
%threshold=[0.6 0.7 0.7]
threshold=[0.6 0.6 0.7 0.94]; 
%scale factor
factor=0.709;
%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel'); 
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir =strcat(caffe_model_path,'/3con2fc.prototxt');
model_dir = strcat(caffe_model_path,'/3con2fc.caffemodel');
claNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
faces=cell(0);	
w_w=fopen('blank2.txt','w');
for i = 1:1
	fold_file = [ 'fold/FDDB-fold-' sprintf('%02d',i) '.txt'];
    write_file=['detection4cifar/foldtt-' sprintf('%02d',i) '-out.txt'];
    f_file = fopen(write_file, 'w');
	A = importdata(fold_file);
    for j = 1:length(A)
        img_fname = A{j};
        img_name=[img_fname '.jpg'];
        img = imread(img_name);
        chn=size(img,3);
        if chn==1
           fprintf(w_w,'%s\n',img_fname); 
           fprintf(f_file,'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa %d\n',0);
           
           continue
        end
        tic
        [boudingboxes points]=detect_facecifar(img,minsize,PNet,claNet,RNet,ONet,threshold,false,factor);
        toc
        faces{i,1}={boudingboxes};
        faces{i,2}={points'};
        %show detection result
        numbox=size(boudingboxes,1);
        fprintf(f_file,'%s\n',img_fname);
        fprintf(f_file,'%d\n',numbox);
        for jj=1:numbox
            w=boudingboxes(jj,3)-boudingboxes(jj,1);
            h=boudingboxes(jj,4)-boudingboxes(jj,2);
            fprintf(f_file,'%f %f %f %f %f\n',boudingboxes(jj,1),boudingboxes(jj,2),w,h,boudingboxes(jj,5));
        end
        
    end
    fclose(f_file);	
end
fclose(w_w);
