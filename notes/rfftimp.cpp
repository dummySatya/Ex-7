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

int main()
{

    std::vector<std::complex<float>> input_tensor_values_complex = readAudioFile("../audio/small.wav");
    int n = input_tensor_values_complex.size();
    std::cout << n << "\n";

    std::vector<double> input_tensor_values(n);
    for (int i = 0; i < n; i++)
    {
        input_tensor_values[i] = input_tensor_values_complex[i].real();
    }

    std::vector<std::complex<float>> fftval = rfft(input_tensor_values_complex);

    std::vector<float> rfftMagnitude = rfft_mag(fftval);

    for (int i = 0; i < 10; i++)
    {
        std::cout << rfftMagnitude[i] << "\n";
    }
}