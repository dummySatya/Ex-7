#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <sndfile.h>
#include <vector>
#include <samplerate.h>
#include <chrono>
#include <thread>
#include <complex>

int nextPowerOf2(int n)
{
    if (n <= 1)
        return 1;
    return std::pow(2, std::ceil(std::log2(n)));
}

std::vector<std::complex<float>> readAudioFile(const char *filename)
{
    SF_INFO sfinfo;
    SNDFILE *infile = sf_open(filename, SFM_READ, &sfinfo);
    if (!infile)
    {
        std::cerr << "Error opening file: " << sf_strerror(nullptr) << std::endl;
        return {};
    }

    // Read the audio data into a vector
    std::vector<double> samples(sfinfo.frames * sfinfo.channels);
    sf_readf_double(infile, samples.data(), sfinfo.frames);
    sf_close(infile);

    // Convert to mono if stereo (average the channels)
    std::vector<double> monoSamples;
    if (sfinfo.channels > 1)
    {
        monoSamples.resize(sfinfo.frames);
        for (int i = 0; i < sfinfo.frames; ++i)
        {
            monoSamples[i] = (samples[i * 2] + samples[i * 2 + 1]) / 2.0f;
        }
    }
    else
    {
        monoSamples = samples; // Already mono
    }

    int nextPow2 = nextPowerOf2(monoSamples.size());

    monoSamples.resize(nextPow2, 0.0f);

    std::vector<std::complex<float>> complexSamples(nextPow2);
    for (int i = 0; i < nextPow2; ++i)
    {
        complexSamples[i] = std::complex<float>(monoSamples[i], 0.0);
    }
    return complexSamples;
}

std::vector<std::complex<float>> rfft(std::vector<std::complex<float>> &samples)
{
    int N = samples.size();
    if (N == 1)
    {
        return samples;
    }
    int M = N / 2;

    std::vector<std::complex<float>> Xeven(M, 0);
    std::vector<std::complex<float>> Xodd(M, 0);

    for (int i = 0; i < M; i++)
    {
        Xeven[i] = samples[2 * i];
        Xodd[i] = samples[2 * i + 1];
    }

    std::vector<std::complex<float>> Feven(M, 0);
    Feven = rfft(Xeven);
    std::vector<std::complex<float>> Fodd(M, 0);
    Fodd = rfft(Xodd);

    std::vector<std::complex<float>> freqbins(N, 0);
    for (int k = 0; k < M; k++)
    {
        std::complex<float>exp = static_cast<std::complex<float>>(std::polar(1.0, -2 * M_PI * k / N));
        std::complex<float> cmplx = exp * Fodd[k];
        freqbins[k] = Feven[k] + cmplx;
        freqbins[k + N / 2] = Feven[k] - cmplx;
    }
    return freqbins;
}

std::vector<float> rfft_mag(std::vector<std::complex<float>> &fft_vals)
{
    int n = fft_vals.size();
    std::vector<float> rfftMag(n / 2 + 1);
    for (int i = 0; i <= n / 2; i++)
    {
        rfftMag[i] = abs(fft_vals[i]);
    }
    return rfftMag;
}

std::vector<float> rfft_onnx(std::vector<float> &input_tensor_values)
{
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "ONNXModel");

    // Create session options
    Ort::SessionOptions session_options;
    // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(6);
    session_options.EnableProfiling("prof/onnx_profile");

    // Create session
    Ort::Session session(env, "./rfft_model.onnx", session_options);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    int inp_shape = input_tensor_values.size();

    std::vector<int64_t> input_shape = {inp_shape};

    // Create input tensor with allocated buffer
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size());

    // Output tensor
    int out_shape = inp_shape % 2 == 0 ? inp_shape / 2 + 1 : (inp_shape + 1) / 2;

    std::vector<float> output_tensor_values(out_shape);

    std::vector<int64_t> output_shape = {out_shape};

    // Create output tensor with allocated buffer
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        output_tensor_values.data(),
        output_tensor_values.size(),
        output_shape.data(),
        output_shape.size());

    // Input and output names (found from Netron)
    const char *input_names[] = {"l_audio_"};
    const char *output_names[] = {"abs_1"};

    auto start = std::chrono::steady_clock::now();

    // Run the model
    session.Run(Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, &output_tensor, 1);

    // Retrieve output data
    float *output_arr = output_tensor.GetTensorMutableData<float>();

    auto end = std::chrono::steady_clock::now();

    // Calculate duration of execution
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Execution time : " << duration << std::endl;
    
    return output_tensor_values;
}

int main()
{

    std::vector<std::complex<float>> input_tensor_values_complex = readAudioFile("../audio/small.wav");
    int n = input_tensor_values_complex.size();
    std::cout << n << "\n";

    std::vector<float> input_tensor_values(n);
    for (int i = 0; i < n; i++)
    {
        input_tensor_values[i] = input_tensor_values_complex[i].real();
    }

    std::vector<std::complex<float>> fftval = rfft(input_tensor_values_complex);

    std::vector<float> rfftMagnitudeCpp = rfft_mag(fftval);
    std::vector<float> rfftMagnitudePython = rfft_onnx(input_tensor_values);

    float diff = 0;
    for(int i = 0; i < n/2;i++){
        diff += rfftMagnitudeCpp[i] - rfftMagnitudePython[i];
    }
    diff = diff / (n/2);

    std::cout<<"Mean Difference: "<<diff<<std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << rfftMagnitudeCpp[i] <<" "<<rfftMagnitudePython[i]<< "\n";
    }
}