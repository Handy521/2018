clear;

%minimum size of face
minsize=40;

%path of toolbox
caffe_model_path='./model';

%use cpu
%caffe.set_mode_cpu();
gpu_id=0;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);

%three steps's threshold
%threshold=[0.6 0.7 0.7]
threshold=[0.6 0.6 0.7]; 
%scale factor
factor=0.709;

%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel'); 
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
faces=cell(0);	
for i = 1:10
	fold_file = [ 'fold/FDDB-fold-' sprintf('%02d',i) '.txt'];
    write_file=['detection/fold-' sprintf('%02d',i) '-out.txt'];
    f_file = open(write_file_name, 'w');
	A = importdata(fold_file);
    for j = 1:length(A)
		img_fname = A{j};
        
	img=imread(imglist{i});
	%we recommend you to set minsize as x * short side
	%minl=min([size(img,1) size(img,2)]);
	%minsize=fix(minl*0.1)
    tic
    [boudingboxes points]=detect_face(img,minsize,PNet,RNet,ONet,threshold,false,factor);
	toc
    faces{i,1}={boudingboxes};
	faces{i,2}={points'};
	%show detection result
	numbox=size(boudingboxes,1);
	imshow(img)
	hold on; 
	for j=1:numbox
		plot(points(1:5,j),points(6:10,j),'g.','MarkerSize',10);
         plot(points(1:2,j),points(6:7,j),'LineWidth',3);
		r=rectangle('Position',[boudingboxes(j,1:2) boudingboxes(j,3:4)-boudingboxes(j,1:2)],'Edgecolor','g','LineWidth',3);
    end
   
    hold off; 
	pause
    end
end
%save result box landmark