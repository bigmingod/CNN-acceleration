#include <vector>

#include "caffe/layers/demo_layer.hpp"

namespace caffe {

template <typename Dtype>
void DemoLayer<Dtype>::compute_output_shape() {
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
void DemoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  /////////////////////////////////////////
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  int count = this->blobs_[0]->count();
  for(int i=0;i<count;++i){
    if(this->masks[i])
      muweight[i]=this->centoids[this->indices[i]];

  }
  /////////////////////////////////////////
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
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
void DemoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  /**/
  int count = this->blobs_[0]->count();
  /**/
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

        /*
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        */
        Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
        for(int j=0;j<count;++j){
          weight_diff[j] *= this->masks[j];
        }
        vector<Dtype> tempdiff(256);
        vector<int> freq(256);
        for(int j=0;j<count;++j){
          if(this->masks[j]){
            tempdiff[this->indices[j]] += weight_diff[j];
            freq[this->indices[j]]++;
          }
        }
        for(int j=0;j<count;j++){
          if(this->masks[j]){
            weight_diff[j] = tempdiff[this->indices[j]] / freq[this->indices[j]];
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}



///////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void DemoLayer<Dtype>::Compute_Blob_masks(float ratio){
  printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

  int count = this->blobs_[0]->count();
  this->masks.resize(count);
  this->indices.resize(count);
  this->centoids.resize(256);

  const Dtype* weight = this->blobs_[0]->cpu_data();

  int index = (int)count * ratio;

  vector<int> sort_weight(count);

  for (int i=0;i<count;i++){
    sort_weight[i] = fabs(weight[i]);
  }
  sort(sort_weight.begin(),sort_weight.end());


  Dtype thr;
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  float rat = 0;
  if (index > 0){

    thr = sort_weight[index-1];

    for(int i=0;i<count;i++){
      this->masks[i]=(((weight[i] > thr)||(weight[i] < -thr)) ? 1:0);
      muweight[i] *= this -> masks[i];
      rat += (1-this->masks[i]);
    }

  }
  else {
    for (int i=0;i<count;i++){
      this->masks[i] = (weight[i]==0?0:1);
      rat += (1-this->masks[i]);
      
    }
  }
  
  kmeans_cluster(this->indices,this->centoids,muweight,count,this->masks,256,1000);

}



//////////////////////////////////////////////////////////////////////////



#ifdef CPU_ONLY
STUB_GPU(DemoLayer);
#endif

INSTANTIATE_CLASS(DemoLayer);

}  // namespace caffe
