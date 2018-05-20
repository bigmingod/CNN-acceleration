#include <vector>
#include <cstdio>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}
template <typename Dtype>
void ConvolutionLayer<Dtype>::ComputeBlobMask()
{
  int count = this->blobs_[0]->count();
  this->masks_.resize(count);
  this->totaldiff_.resize(this->blobs_[0]->channels());
  this->whennow = 0;
  for(int i=0;i<count;i++) this->masks_[i] = 1;
  for(int i=0;i<this->blobs_[0]->channels();i++) this->totaldiff_[i] = 0;
  this->inprune = true;
  this->testcnt = 0;
  int thenum=this->blobs_[0]->num(),thechannels=this->blobs_[0]->channels(),theh=this->blobs_[0]->height(),thew=this->blobs_[0]->width();
  this->testblob = new Blob<Dtype>(this->blobs_[0]->num(),this->blobs_[0]->channels(),this->blobs_[0]->height(),this->blobs_[0]->width());
  Dtype* mutest = this->testblob->mutable_cpu_data();
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  for(int i=0;i<thenum;i++)
    for(int j=0;j<thechannels;j++)
      for(int k=0;k<theh;k++)
        for(int l=0;l<thew;l++)
          mutest[this->testblob->offset(i,j,k,l)] = muweight[this->blobs_[0]->offset(i,j,k,l)];
}
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int thenum=this->blobs_[0]->num(),thechannels=this->blobs_[0]->channels(),theh=this->blobs_[0]->height(),thew=this->blobs_[0]->width();
  Dtype* weight = this->blobs_[0]->mutable_cpu_data(); 
  if(this->testcnt > 100000&&this->inprune)
  {
    this->inprune = false;
    bool b[thechannels];
    for(int i=0;i<thechannels;i++) b[i]=false;
    int now=0;
    while(now<thechannels/2)
    {
      now++;
      Dtype minn = -1;
      int pos = 0;
      for(int i=0;i<thechannels;i++) if (b[i]==false)
      {
        if (this->totaldiff_[i]<minn||(minn==-1))
        {
          minn = this->totaldiff_[i];
          pos = i;
        }
      }
      b[pos]=true;
      for(int i=0;i<thenum;i++)
        for(int j=0;j<theh;j++)
          for(int k=0;k<thew;k++)
          {
            weight[this->blobs_[0]->offset(i,pos,j,k)] = 0;
            this->masks_[this->blobs_[0]->offset(i,pos,j,k)] = 0;
          }
      this->prunedone = true;
    }
  }
  this->testblob = new Blob<Dtype>(this->blobs_[0]->num(),this->blobs_[0]->channels(),this->blobs_[0]->height(),this->blobs_[0]->width());
  Dtype* mutest = this->testblob->mutable_cpu_data();
  for(int i=0;i<thenum;i++)
    for(int j=0;j<thechannels;j++)
      for(int k=0;k<theh;k++)
        for(int l=0;l<thew;l++)
          mutest[this->testblob->offset(i,j,k,l)] = weight[this->blobs_[0]->offset(i,j,k,l)];
  //printf("!%d\n",this->testcnt);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    int rand_key = rand()%5;
    if(this->inprune&&rand_key==1)
    {
      for (int j=0;j<thechannels;j++)
      {
        for(int ii=0;ii<thenum;ii++)
            for(int jj=0;jj<theh;jj++)
              for(int kk=0;kk<thew;kk++)  mutest[this->testblob->offset(ii,j,jj,kk)] *= 0;
        for(int n=0;n<this->num_;++n)
        {
          this->forward_cpu_gemm(bottom_data + n*this->bottom_dim_,mutest,top_data);
          this->forward_cpu_gemm(bottom_data + n*this->bottom_dim_,weight,top_data+this->top_dim_);
          for (int ii=0;ii<10;ii++)
          {
            this->testcnt += 1;
            int ww=this->output_shape_[1];
            int rh=rand()%this->output_shape_[0];
            int rw=rand()%this->output_shape_[1];
            Dtype add_sum = top_data[this->top_dim_+rh*ww+rw]-top_data[rh*ww+rw];
            add_sum = add_sum * add_sum;
            this->totaldiff_[j] += add_sum;
          }
        }
        for(int ii=0;ii<thenum;ii++)
            for(int jj=0;jj<theh;jj++)
              for(int kk=0;kk<thew;kk++)  mutest[this->testblob->offset(ii,j,jj,kk)] = weight[this->blobs_[0]->offset(ii,j,jj,kk)];
      }
    }
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  int count = this->blobs_[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        if(this->prunedone) for (int j=0;j<count;j++) weight_diff[j]*=this->masks_[j];
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
