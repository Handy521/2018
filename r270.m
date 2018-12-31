clear;caffe.reset_all();%释放内存
minsize=50;
caffe_path='H:/caffe-master/Build/x64/Release/matcaffe';
pdollar_toolbox_path='H:/bookandvideo/数字图像处理（冈萨雷斯）/toolbox-master';
caffe_model_path='./model';
addpath(genpath(caffe_path));
addpath(genpath(pdollar_toolbox_path));
gpu_id=0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
threshold=[0.6 0.7 0.7];
factor=0.709;
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir =strcat(caffe_model_path,'/32fc_deploy.prototxt');
model_dir = strcat(caffe_model_path,'/4_120000.caffemodel');
claNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');

factor_count=0;
total_boxes=[];
img1=imread('photo/1.jpg');
h=size(img1,1);
w=size(img1,2);
minl=min([w h]);
img=single(img1);

m=12/minsize;
minl=minl*m;
%creat scale pyramid
scales=[];
while (minl>=12)
    scales=[scales m*factor^(factor_count)];
    minl=minl*factor;
    factor_count=factor_count+1;
end
%first stage
for j = 1:size(scales,2)
    scale=scales(j);
    hs=ceil(h*scale);
    ws=ceil(w*scale);
  

    im_data=(imResample(img,[hs ws],'bilinear')-127.5)*0.0078125;
   
    PNet.blobs('data').reshape([hs ws 3 1]);
    out=PNet.forward({im_data});
    boxes=generateBoundingBox(out{2}(:,:,2),out{1},scale,threshold(1));
    %inter-scale nms
    pick=nms(boxes,0.5,'Union');
    boxes=boxes(pick,:);
    if ~isempty(boxes)
        total_boxes=[total_boxes;boxes];
    end
end
numbox=size(total_boxes,1);
if ~isempty(total_boxes)
    pick=nms(total_boxes,0.7,'Union');
    total_boxes=total_boxes(pick,:);
    regw=total_boxes(:,3)-total_boxes(:,1);
    regh=total_boxes(:,4)-total_boxes(:,2);
    total_boxes=[total_boxes(:,1)+total_boxes(:,6).*regw total_boxes(:,2)+total_boxes(:,7).*regh total_boxes(:,3)+total_boxes(:,8).*regw total_boxes(:,4)+total_boxes(:,9).*regh total_boxes(:,5)];
    total_boxes=rerec(total_boxes);
    total_boxes(:,1:4)=fix(total_boxes(:,1:4));
    [dy edy dx edx y ey x ex tmpw tmph]=pad(total_boxes,w,h);
end
numbox=size(total_boxes,1);
if numbox>0
    %second stage
    tempimg=zeros(32,32,3,numbox);
    for k=1:numbox
        tmp=zeros(tmph(k),tmpw(k),3);
        tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
        tmp=tmp(:,:,[3,2,1]);%RGB to BGR
        tmp=permute(tmp,[2,1,3]);
        tempimg(:,:,:,k)=imResample(tmp,[32 32],'bilinear');
    end
   
    claNet.blobs('data').reshape([32 32 3 numbox]);
    claNet.forward({tempimg});
    feat=claNet.blobs('prob').get_data();
    a=[1,2,3,4];
    b=zeros(1,numbox);
    for ii=1:numbox
        for jj=1:4
            a(jj)=feat(:,:,jj,ii);
        end
        [xx,yy]=max(a);
        b(ii)=yy;
    end
    total_boxes=[total_boxes,b'];
    rotate=find(total_boxes(:,6)>1);
    save_rotate=total_boxes(rotate,:); %4分类判断
    numbox=size( save_rotate,1);
		if numbox>0
            [rdy redy rdx redx ry rey rx rex rtmpw rtmph]=pad(save_rotate,w,h);
            rtempimg=zeros(48,48,3,3);
			for k=1:3
				rtmp=zeros(rtmph(k),rtmpw(k),3);
				rtmp(rdy(k):redy(k),rdx(k):redx(k),:)=img(ry(k):rey(k),rx(k):rex(k),:);
                rtmp=imrotate(rtmp,270);%对框中内容旋转270
				rtempimg(:,:,:,k)=imResample(rtmp,[48 48],'bilinear');
            end
            rtempimg=(rtempimg-127.5)*0.0078125;
			ONet.blobs('data').reshape([48 48 3 3]);
			out=ONet.forward({rtempimg});
			score=squeeze(out{3}(2,:));
			points=out{2};
			pass=find(score>threshold(3));
			points=points(:,pass);
			total_boxes=[save_rotate(pass,1:4) score(pass)' ];
			mv=out{1}(:,pass);                          
			if size(total_boxes,1)>0		
                w1=total_boxes(:,3)-total_boxes(:,1)+1;
                h1=total_boxes(:,4)-total_boxes(:,2)+1;
                numP=size(points,2);
                c=reshape(points,1,10*numP);                
                jj=0;
                for i=0:5:5*numP-1
                    dd(1,:)=c(1+2*i:5+2*i);
                    dd(2,:)=c(6+2*i:10+2*i);
                    dd(3,:)=1;            
                    M=[1   0    0.5;
                        0   1   0.5;
                        0    0    1  ];
                    M1=[cos(3*pi/2)   sin(3*pi/2)   0;%旋转270度
                        -sin(3*pi/2)   cos(3*pi/2)  0;
                        0           0         1];
                    M2=[1    0    -0.5;
                        0    1    -0.5;
                        0    0       1  ];
                    mm=M*M1*M2;  
                    cc(:,1+i:5+i)  =mm*dd;
                    jj=jj+1;
                    points(1:5,jj)=cc(1,1+i:5+i);
                    points(6:10,jj)=cc(2,1+i:5+i);
                end
               
                points(1:5,:)=repmat(total_boxes(:,3)',[5 1])-repmat(w1',[5 1]).*points(1:5,:)+1;
                points(6:10,:)=repmat(total_boxes(:,4)',[5 1])-repmat(h1',[5 1]).*points(6:10,:)+1;    
                
				total_boxes=bbre270(total_boxes,mv(:,:)');	               
                pick=nms(total_boxes,0.7,'Min');
				total_boxes=total_boxes(pick,:);  	
              
                points=points(:,pick);
                
			end
		end
end
numbox=size(total_boxes,1);
	imshow(img1)
	hold on; 
	for j=1:numbox
		plot(points(1:5,j),points(6:10,j),'g.','MarkerSize',10);
         plot(points(1:2,j),points(6:7,j),'LineWidth',3);
		r=rectangle('Position',[total_boxes(j,1:2) total_boxes(j,3:4)-total_boxes(j,1:2)],'Edgecolor','g','LineWidth',3);
       % rectangle('Position',[total_boxes2(j,1:2) total_boxes2(j,3:4)-total_boxes(j,1:2)],'Edgecolor','g','LineWidth',3);
    end

tic
caffe.reset_all();%释放内存
toc