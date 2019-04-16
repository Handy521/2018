clear;caffe.reset_all();


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


    
	img=imread('rotate/test18.jpg');
	%we recommend you to set minsize as x * short side
	%minl=min([size(img,1) size(img,2)]);
	%minsize=fix(minl*0.1)
    tic
    [boudingboxes points]=detect_facecifar(img,minsize,PNet,claNet,RNet,ONet,threshold,false,factor);
	toc

	%show detection result
	numbox=size(boudingboxes,1);
	imshow(img)
	hold on; 
    for  j=1:numbox
        plot(points(1:5,j),points(6:10,j),'g.','MarkerSize',10);
        plot(points(1:2,j),points(6:7,j),'LineWidth',3);
        r=rectangle('Position',[boudingboxes(j,1:2) boudingboxes(j,3:4)-boudingboxes(j,1:2)],'Edgecolor','g','LineWidth',3);%x y w h
    end
   
    hold off; 


%save result box landmark