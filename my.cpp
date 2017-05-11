#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
    public:
        Classifier(const string& model_file,
                const string& trained_file,
                const string& mean_file,
                const string& label_file,
                const bool use_GPU);

        std::vector<std::vector<Prediction> > Classify(const std::vector<cv::Mat>& imgs, int N = 5);
        private:
        void SetMean(const string& mean_file);

        std::vector<std::vector<float> > Predict(const std::vector<cv::Mat>& imgs);

        void WrapInputLayer(std::vector<cv::Mat>* input_channels, int n);

        void Preprocess(const cv::Mat& img,
        std::vector<cv::Mat>* input_channels);

    private:
        shared_ptr<Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat mean_;
        std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                    const string& trained_file,
                    const string& mean_file,
                    const string& label_file,
                    const bool use_GPU) {
    if (use_GPU)        
		Caffe::set_mode(Caffe::GPU);    
	else        
		Caffe::set_mode(Caffe::CPU);

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file);

    /* Load labels. */
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line))
    labels_.push_back(string(line));

    Blob<float>* output_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), output_layer->channels()) << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

/* Return the top N predictions. */
std::vector<std::vector<Prediction> > Classifier::Classify(const std::vector<cv::Mat>& imgs, int N) {
    std::vector<std::vector<float> > outputs = Predict(imgs);

    std::vector<std::vector<Prediction> > all_predictions;
    for (int j = 0; j < outputs.size(); ++j) {
        std::vector<float> output = outputs[j];

        N = std::min<int>(labels_.size(), N);
        std::vector<int> maxN = Argmax(output, N);
        std::vector<Prediction> predictions;
        for (int i = 0; i < N; ++i) {
            int idx = maxN[i];
            predictions.push_back(std::make_pair(labels_[idx], output[idx]));
            }
        all_predictions.push_back(predictions);
    }

    return all_predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_) << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
    * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<std::vector<float> > Classifier::Predict(const std::vector<cv::Mat>& imgs) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(imgs.size(), num_channels_,
    input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    for (int i = 0; i < imgs.size(); ++i) {
        std::vector<cv::Mat> input_channels;
        WrapInputLayer(&input_channels, i);
        Preprocess(imgs[i], &input_channels);
    }

    net_->Forward();

    /* Copy the output layer to a std::vector */
    std::vector<std::vector<float> > outputs;

    Blob<float>* output_layer = net_->output_blobs()[0];
    for (int i = 0; i < output_layer->num(); ++i) {
        const float* begin = output_layer->cpu_data() + i * output_layer->channels();
        const float* end = begin + output_layer->channels();
        /* Copy the output layer to a std::vector */
        outputs.push_back(std::vector<float>(begin, end));
    }
    return outputs;
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels, int n) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int channels = input_layer->channels();
    float* input_data = input_layer->mutable_cpu_data() + n * width * height * channels;

    for (int i = 0; i < channels; ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    /*
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
    == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
    */
}

#define MAX_PATH_LENGTH     512

int main(int argc, char** argv) {
    if (argc != 8) {
		std::cerr << "ERROR, the argv is not 8" << std::endl;
        std::cerr << "Usage: " << "classification.bin"
        << " deploy.prototxt network.caffemodel"
        << " mean.binaryproto labels.txt batch_size image work_mode(1:GPU;0:CPU)" << std::endl;
        return 1;
    }
    DIR         *ptr_dir;
    struct dirent   *dir_entry;
    char        *file_path;
    int fileNum = 0;
	int batchNum = 0;

    ::google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = 1;

    string model_file   = argv[1];
    string trained_file = argv[2];
    string mean_file    = argv[3];
    string label_file   = argv[4];
    int ibatch_size = atoi(argv[5]);
	long int predict_picture_use = 0;
	long int load_model_use = 0;
	long int read_picture_use = 0;
	struct timeval start;	
	struct timeval end;

	gettimeofday(&start,NULL);
    Classifier classifier(model_file, trained_file, mean_file, label_file, atoi(argv[7]));
	gettimeofday(&end,NULL);
	load_model_use = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);

    std::vector<cv::Mat> imgs;
	std::vector<cv::Mat> imgs_tmp;

    file_path = (char*)malloc(sizeof(char)*MAX_PATH_LENGTH);
    if(file_path == NULL)
	{
        printf("allocate memory for file path failed.\n");
        return 1;
    }
    memset(file_path, 0, sizeof(char)*MAX_PATH_LENGTH);

    ptr_dir = opendir(argv[6]);

	gettimeofday(&start,NULL);
    while((dir_entry = readdir(ptr_dir)) != NULL)
	{
        if(dir_entry->d_type & DT_REG)
		{
            sprintf(file_path, "%s%s", argv[6], dir_entry->d_name);
            //std::cout << file_path << std::endl;
            cv::Mat img = cv::imread(file_path, -1);
            imgs.push_back(img);
            fileNum++;
        }
    }
	gettimeofday(&end,NULL);
	read_picture_use = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);


	gettimeofday(&start,NULL);
    LOG(ERROR);
    for (int i = 0; i < imgs.size(); i++)
    {

		imgs_tmp.push_back(imgs[i]);

		if (imgs_tmp.size() == ibatch_size)
		{
		//printf("imgs_tmp.size() :%d,line 310 \r\n",(int)imgs_tmp.size());
			//LOG(ERROR);
	        std::vector<std::vector<Prediction> > all_predictions = classifier.Classify(imgs_tmp);
		//LOG(ERROR);
	        /* Print the top N predictions. */
	        /*for (size_t i = 0; i < all_predictions.size(); ++i) 
			{
	            std::cout << "---------- Prediction for " << file_path[i] << " ----------" << std::endl;
	            std::vector<Prediction>& predictions = all_predictions[i];

	            for (size_t j = 0; j < predictions.size(); ++j) {
	                Prediction p = predictions[j];
	                std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
	            }
	            std::cout << std::endl;
	        }*/

	        imgs_tmp.clear();
			batchNum++;
			
		}
		
    }

	if (imgs_tmp.size() != 0)
	{
	//printf("imgs_tmp.size() :%d,line 334 \r\n",(int)imgs_tmp.size());

        std::vector<std::vector<Prediction> > all_predictions = classifier.Classify(imgs_tmp);

        /* Print the top N predictions. */
        /*for (size_t i = 0; i < all_predictions.size(); ++i) 
		{
            std::cout << "---------- Prediction for " << file_path[i] << " ----------" << std::endl;
            std::vector<Prediction>& predictions = all_predictions[i];

            for (size_t j = 0; j < predictions.size(); ++j) {
                Prediction p = predictions[j];
                std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
            }
            std::cout << std::endl;
        }*/
        //fileNum = 0;
        imgs_tmp.clear();

		batchNum++;
		
	}
	LOG(ERROR);
	gettimeofday(&end,NULL);
	predict_picture_use = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);

	std::cout << "batch_size:  " << ibatch_size << std::endl;
	std::cout << "batch_number:" << batchNum << std::endl;
	std::cout << "fileNumber:  " << fileNum << std::endl;
	printf("load model   time:%ld us\r\n",load_model_use);
	printf("read picture time:%ld us\r\n",read_picture_use);
	printf("predict      time:%ld us\r\n",predict_picture_use);
    
    free(file_path);
    file_path = NULL;

}

