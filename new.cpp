#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TMath.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

using std::cout;
using std::endl;
using namespace std;

// Constants
const int N_PMTS = 12;
const int PMT_CHANNEL_MAP[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const int PULSE_THRESHOLD = 30;     // ADC threshold for pulse detection
const int BS_UNCERTAINTY = 5;       // Baseline uncertainty (ADC)
const int EV61_THRESHOLD = 1200;    // Beam on if channel 22 > this (ADC)
const double MUON_ENERGY_THRESHOLD = 20; // Min PMT energy for muon (p.e.)
const double MICHEL_ENERGY_MIN = 40;    // Min PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX = 1000;  // Max PMT energy for Michel (p.e.)
const double MICHEL_DT_MIN = 1.1;       // Min time after muon for Michel (µs)
const double MICHEL_DT_MAX = 16.0;      // Max time after muon for Michel (µs)
const int ADCSIZE = 45;                 // Number of ADC samples per waveform
//const int MAX_NUM_ENTRIES = 1900000;    // Max waveforms to process
const string OUTPUT_DIR = "AnalysisOutput";
const string OUTPUT_STATS_NAME = OUTPUT_DIR + "/PMTAnalysisStats.txt";
const std::vector<double> SIDE_VP_THRESHOLDS = {750, 950, 1200, 1375, 525, 700, 700, 500}; // Channels 12-19 (ADC)
const double TOP_VP_THRESHOLD = 450; // Channels 20-21 (ADC)
const double FIT_MIN = 1.1; // Fit range min (µs)
const double FIT_MAX = 16.0; // Fit range max (µs)

// Pulse structure
struct pulse {
    double start;          // Start time (µs)
    double end;            // End time (µs)
    double peak;           // Max amplitude (p.e. for PMTs, ADC for SiPMs)
    double energy;         // Energy (p.e. for PMTs, ADC for SiPMs)
    double number;         // Number of channels with pulse
    bool single;           // Timing consistency
    bool beam;             // Beam status
    double trigger;        // Trigger type
    double side_vp_energy; // Side veto energy (ADC)
    double top_vp_energy;  // Top veto energy (ADC)
    double all_vp_energy;  // All veto energy (ADC)
    double last_muon_time; // Time of last muon (µs)
    bool is_muon;          // Muon candidate flag
    bool is_michel;        // Michel electron candidate flag
};

// Temporary pulse structure
struct pulse_temp {
    double start;  // Start time (µs)
    double end;    // End time (µs)
    double peak;   // Max amplitude
    double energy; // Energy
};

// SPE fitting function
Double_t SPEfit(Double_t *x, Double_t *par) {
    Double_t term1 = par[0] * exp(-0.5 * pow((x[0] - par[1]) / par[2], 2));
    Double_t term2 = par[3] * exp(-0.5 * pow((x[0] - par[4]) / par[5], 2));
    Double_t term3 = par[6] * exp(-0.5 * pow((x[0] - sqrt(2) * par[4]) / sqrt(2 * pow(par[5], 2) - pow(par[2], 2)), 2));
    Double_t term4 = par[7] * exp(-0.5 * pow((x[0] - sqrt(3) * par[4]) / sqrt(3 * pow(par[5], 2) - 2 * pow(par[2], 2)), 2));
    return term1 + term2 + term3 + term4;
}

// Exponential fit function: N0 * exp(-t/tau) + C (t, tau in µs)
Double_t ExpFit(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0] / par[1]) + par[2];
}

// Utility functions
template<typename T>
double getAverage(const std::vector<T>& v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template<typename T>
double mostFrequent(const std::vector<T>& v) {
    if (v.empty()) return 0;
    std::map<T, int> count;
    for (const auto& val : v) count[val]++;
    T most_common = v[0];
    int max_count = 0;
    for (const auto& pair : count) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_common = pair.first;
        }
    }
    return max_count > 1 ? most_common : getAverage(v);
}

template<typename T>
double variance(const std::vector<T>& v) {
    if (v.size() <= 1) return 0;
    double mean = getAverage(v);
    double sum = 0;
    for (const auto& val : v) {
        sum += (val - mean) * (val - mean);
    }
    return sum / (v.size() - 1);
}

// Create output directory
void createOutputDirectory(const string& dirName) {
    struct stat st;
    if (stat(dirName.c_str(), &st) != 0) {
        if (mkdir(dirName.c_str(), 0755) != 0) {
            cerr << "Error: Could not create directory " << dirName << endl;
            exit(1);
        }
        cout << "Created output directory: " << dirName << endl;
    } else {
        cout << "Output directory already exists: " << dirName << endl;
    }
}

// SPE calibration function
void performCalibration(const string &calibFileName, Double_t *mu1, Double_t *mu1_err) {
    TFile *calibFile = TFile::Open(calibFileName.c_str());
    if (!calibFile || calibFile->IsZombie()) {
        cerr << "Error opening calibration file: " << calibFileName << endl;
        exit(1);
    }

    TTree *calibTree = (TTree*)calibFile->Get("tree");
    if (!calibTree) {
        cerr << "Error accessing tree in calibration file" << endl;
        calibFile->Close();
        exit(1);
    }

    TCanvas *c = new TCanvas("c", "SPE Fits", 800, 600);
    TH1F *histArea[N_PMTS];
    Long64_t nLEDFlashes[N_PMTS] = {0};
    for (int i = 0; i < N_PMTS; i++) {
        histArea[i] = new TH1F(Form("PMT%d_Area", i + 1),
                               Form("PMT %d;ADC Counts;Events", i + 1), 150, -50, 400);
    }

    Int_t triggerBits;
    Double_t area[23];
    calibTree->SetBranchAddress("triggerBits", &triggerBits);
    calibTree->SetBranchAddress("area", area);

    Long64_t nEntries = calibTree->GetEntries();
    cout << "Processing " << nEntries << " calibration events from " << calibFileName << "..." << endl;

    for (Long64_t entry = 0; entry < nEntries; entry++) {
        calibTree->GetEntry(entry);
        if (triggerBits != 16) continue;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            histArea[pmt]->Fill(area[PMT_CHANNEL_MAP[pmt]]);
            nLEDFlashes[pmt]++;
        }
    }

    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]->GetEntries() < 1000) {
            cerr << "Warning: Insufficient data for PMT " << i + 1 << " in " << calibFileName << endl;
            mu1[i] = 0;
            mu1_err[i] = 0;
            delete histArea[i];
            continue;
        }

        TF1 *fitFunc = new TF1("fitFunc", SPEfit, -50, 400, 8);
        Double_t histMean = histArea[i]->GetMean();
        Double_t histRMS = histArea[i]->GetRMS();

        fitFunc->SetParameters(1000, histMean - histRMS, histRMS / 2,
                              1000, histMean, histRMS,
                              500, 200);

        histArea[i]->Fit(fitFunc, "Q", "", -50, 400);

        mu1[i] = fitFunc->GetParameter(4);
        Double_t sigma_mu1 = fitFunc->GetParError(4);
        Double_t sigma1 = fitFunc->GetParameter(5);
        mu1_err[i] = sqrt(pow(sigma_mu1, 2) + pow(sigma1 / sqrt(nLEDFlashes[i]), 2));

        // Plot SPE fit
        c->Clear();
        histArea[i]->Draw();
        fitFunc->Draw("same");
        TLegend *leg = new TLegend(0.6, 0.7, 0.9, 0.9);
        leg->AddEntry(histArea[i], Form("PMT %d Data", i + 1), "l");
        leg->AddEntry(fitFunc, "SPE Fit", "l");
        leg->AddEntry((TObject*)0, Form("mu1 = %.2f #pm %.2f", mu1[i], mu1_err[i]), "");
        leg->Draw();
        string plotName = OUTPUT_DIR + Form("/SPE_Fit_PMT%d.png", i + 1);
        c->Update();
        c->SaveAs(plotName.c_str());
        cout << "Saved SPE plot: " << plotName << endl;
        delete leg;
        delete fitFunc;
        delete histArea[i];
    }

    delete c;
    calibFile->Close();
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <calibration_file> <input_file1> [<input_file2> ...]" << endl;
        return -1;
    }

    string calibFileName = argv[1];
    vector<string> inputFiles;
    for (int i = 2; i < argc; i++) {
        inputFiles.push_back(argv[i]);
    }

    // Create output directory
    createOutputDirectory(OUTPUT_DIR);

    cout << "Calibration file: " << calibFileName << endl;
    cout << "Input files:" << endl;
    for (const auto& file : inputFiles) {
        cout << "  " << file << endl;
    }

    // Check if calibration file exists
    if (gSystem->AccessPathName(calibFileName.c_str())) {
        cerr << "Error: Calibration file " << calibFileName << " not found" << endl;
        return -1;
    }

    // Check if at least one input file exists
    bool anyInputFileExists = false;
    for (const auto& file : inputFiles) {
        if (!gSystem->AccessPathName(file.c_str())) {
            anyInputFileExists = true;
            break;
        }
    }
    if (!anyInputFileExists) {
        cerr << "Error: No input files found" << endl;
        return -1;
    }

    // Perform SPE calibration
    Double_t mu1[N_PMTS] = {0};
    Double_t mu1_err[N_PMTS] = {0};
    performCalibration(calibFileName, mu1, mu1_err);

    // Print calibration results
    cout << "SPE Calibration Results (from " << calibFileName << "):\n";
    for (int i = 0; i < N_PMTS; i++) {
        cout << "PMT " << i + 1 << ": mu1 = " << mu1[i] << " ± " << mu1_err[i] << " ADC counts/p.e.\n";
    }

    // Statistics counters
    int num_muons = 0;
    int num_michels = 0;
    int num_events = 0;

    // Define histograms
    TH1D* h_muon_energy = new TH1D("h_muon_energy", "Muon Energy Distribution (with Michel Electrons);Energy (p.e.);Counts", 100, -500, 2000);
    TH1D* h_michel_energy = new TH1D("h_michel_energy", "Michel Electron Energy Distribution;Energy (p.e.);Counts", 100, 0, 800);
    TH1D* h_dt_michel = new TH1D("h_dt_michel", "Time Difference Muon to Michel Electron;dt (#mus);Counts", 160, 0, MICHEL_DT_MAX);
    TH2D* h_energy_vs_dt = new TH2D("h_energy_vs_dt", "Michel Energy vs Time Difference;dt (#mus);Energy (p.e.)", 160, 0, MICHEL_DT_MAX, 200, 0, 2000);
    TH1D* h_side_vp_muon = new TH1D("h_side_vp_muon", "Side Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 5000);
    TH1D* h_top_vp_muon = new TH1D("h_top_vp_muon", "Top Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 1000);

    // Open stats file
    std::ofstream statsFile(OUTPUT_STATS_NAME, std::ios::out);
    if (!statsFile.is_open()) {
        cerr << "Error: Could not open stats file " << OUTPUT_STATS_NAME << endl;
        return -1;
    }

    for (const auto& inputFileName : inputFiles) {
        // Check if input file exists
        if (gSystem->AccessPathName(inputFileName.c_str())) {
            cout << "Could not open file: " << inputFileName << ". Skipping..." << endl;
            continue;
        }

        TFile *f = new TFile(inputFileName.c_str());
        cout << "Processing file: " << inputFileName << endl;

        TTree* t = (TTree*)f->Get("tree");
        if (!t) {
            cout << "Could not find tree in file: " << inputFileName << endl;
            f->Close();
            continue;
        }

        // Declaration of leaf types
        Int_t eventID;
        Int_t nSamples[23];
        Short_t adcVal[23][45];
        Double_t baselineMean[23];
        Double_t baselineRMS[23];
        Double_t pulseH[23];
        Int_t peakPosition[23];
        Double_t area[23];
        Long64_t nsTime;
        Int_t triggerBits;

        // Set branch addresses
        t->SetBranchAddress("eventID", &eventID);
        t->SetBranchAddress("nSamples", nSamples);
        t->SetBranchAddress("adcVal", adcVal);
        t->SetBranchAddress("baselineMean", baselineMean);
        t->SetBranchAddress("baselineRMS", baselineRMS);
        t->SetBranchAddress("pulseH", pulseH);
        t->SetBranchAddress("peakPosition", peakPosition);
        t->SetBranchAddress("area", area);
        t->SetBranchAddress("nsTime", &nsTime);
        t->SetBranchAddress("triggerBits", &triggerBits);

        // Create output file
        string outputFile = OUTPUT_DIR + "/PMTWaveformAnalysis_" + inputFileName;
        TFile *fileOut = new TFile(outputFile.c_str(), "RECREATE");
        if (!fileOut || fileOut->IsZombie()) {
            cerr << "Error: Could not create output file " << outputFile << endl;
            f->Close();
            continue;
        }
        TTree *eventTree = new TTree("eventTree", "Muon and Michel Electron Events");

        // Tree branches
        int br_eventID;
        double br_start, br_energy, br_peak, br_number, br_side_vp_energy, br_top_vp_energy, br_all_vp_energy;
        double br_last_muon_time, br_dt_michel;
        bool br_is_muon, br_is_michel, br_single, br_beam;
        int br_trigger;

        eventTree->Branch("eventID", &br_eventID, "eventID/I");
        eventTree->Branch("start", &br_start, "start/D");
        eventTree->Branch("energy", &br_energy, "energy/D");
        eventTree->Branch("peak", &br_peak, "peak/D");
        eventTree->Branch("number", &br_number, "number/D");
        eventTree->Branch("side_vp_energy", &br_side_vp_energy, "side_vp_energy/D");
        eventTree->Branch("top_vp_energy", &br_top_vp_energy, "top_vp_energy/D");
        eventTree->Branch("all_vp_energy", &br_all_vp_energy, "all_vp_energy/D");
        eventTree->Branch("last_muon_time", &br_last_muon_time, "last_muon_time/D");
        eventTree->Branch("dt_michel", &br_dt_michel, "dt_michel/D");
        eventTree->Branch("is_muon", &br_is_muon, "is_muon/O");
        eventTree->Branch("is_michel", &br_is_michel, "is_michel/O");
        eventTree->Branch("single", &br_single, "single/O");
        eventTree->Branch("beam", &br_beam, "beam/O");
        eventTree->Branch("trigger", &br_trigger, "trigger/I");

        int numEntries = std::min((int)t->GetEntries(), MAX_NUM_ENTRIES);
        double last_muon_time = 0.0;
        std::set<double> michel_muon_times; // Store muon times associated with Michel electrons (µs)
        std::vector<std::pair<double, double>> muon_candidates; // Store (start, energy) for muons (µs)

        // First pass: Identify Michel electrons and their muon times
        for (int iEnt = 0; iEnt < numEntries; iEnt++) {
            t->GetEntry(iEnt);
            num_events++;

            // Initialize pulse
            struct pulse p;
            p.start = nsTime / 1000.0; // Convert ns to µs
            p.end = nsTime / 1000.0;
            p.peak = 0;
            p.energy = 0;
            p.number = 0;
            p.single = false;
            p.beam = false;
            p.trigger = triggerBits;
            p.side_vp_energy = 0;
            p.top_vp_energy = 0;
            p.all_vp_energy = 0;
            p.last_muon_time = last_muon_time;
            p.is_muon = false;
            p.is_michel = false;

            std::vector<double> all_chan_start, all_chan_end, all_chan_peak, all_chan_energy;
            std::vector<double> side_vp_energy, top_vp_energy;
            std::vector<double> chan_starts_no_outliers;
            TH1D h_wf("h_wf", "Waveform", ADCSIZE, 0, ADCSIZE);

            bool pulse_at_end = false;
            int pulse_at_end_count = 0;
            std::vector<double> veto_energies(10, 0); // Channels 12-21

            for (int iChan = 0; iChan < 23; iChan++) {
                // Fill waveform histogram
                for (int i = 0; i < ADCSIZE; i++) {
                    h_wf.SetBinContent(i + 1, adcVal[iChan][i] - baselineMean[iChan]);
                }

                // Check beam status (channel 22)
                if (iChan == 22) {
                    double ev61_energy = 0;
                    for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                        ev61_energy += h_wf.GetBinContent(iBin);
                    }
                    if (ev61_energy > EV61_THRESHOLD) {
                        p.beam = true;
                    }
                }

                // Pulse detection
                std::vector<pulse_temp> pulses_temp;
                bool onPulse = false;
                int thresholdBin = 0, peakBin = 0;
                double peak = 0, pulseEnergy = 0;
                double allPulseEnergy = 0;

                for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                    double iBinContent = h_wf.GetBinContent(iBin);
                    if (iBin > 15) allPulseEnergy += iBinContent;

                    if (!onPulse && iBinContent >= PULSE_THRESHOLD) {
                        onPulse = true;
                        thresholdBin = iBin;
                        peakBin = iBin;
                        peak = iBinContent;
                        pulseEnergy = iBinContent;
                    } else if (onPulse) {
                        pulseEnergy += iBinContent;
                        if (peak < iBinContent) {
                            peak = iBinContent;
                            peakBin = iBin;
                        }
                        if (iBinContent < BS_UNCERTAINTY || iBin == ADCSIZE) {
                            pulse_temp pt;
                            pt.start = thresholdBin * 16.0 / 1000.0; // Convert ns to µs
                            pt.peak = iChan <= 11 && mu1[iChan] > 0 ? peak / mu1[iChan] : peak; // p.e. for PMTs, ADC for SiPMs
                            pt.end = iBin * 16.0 / 1000.0; // Convert ns to µs
                            for (int j = peakBin - 1; j >= 1 && h_wf.GetBinContent(j) > BS_UNCERTAINTY; j--) {
                                if (h_wf.GetBinContent(j) > peak * 0.1) {
                                    pt.start = j * 16.0 / 1000.0; // Convert ns to µs
                                }
                                pulseEnergy += h_wf.GetBinContent(j);
                            }
                            if (iChan <= 11) { // PMTs
                                pt.energy = mu1[iChan] > 0 ? pulseEnergy / mu1[iChan] : 0; // Convert to p.e.
                                all_chan_start.push_back(pt.start);
                                all_chan_end.push_back(pt.end);
                                all_chan_peak.push_back(pt.peak);
                                all_chan_energy.push_back(pt.energy);
                                if (pt.energy > 1) p.number += 1;
                            }
                            pulses_temp.push_back(pt);
                            peak = 0;
                            peakBin = 0;
                            pulseEnergy = 0;
                            thresholdBin = 0;
                            onPulse = false;
                        }
                    }
                }

                // Store energy for veto panels (ADC)
                if (iChan >= 12 && iChan <= 19) { // Side veto
                    side_vp_energy.push_back(allPulseEnergy);
                    veto_energies[iChan - 12] = allPulseEnergy;
                } else if (iChan >= 20 && iChan <= 21) { // Top veto
                    double factor = (iChan == 20) ? 1.07809 : 1.0;
                    top_vp_energy.push_back(allPulseEnergy * factor);
                    veto_energies[iChan - 12] = allPulseEnergy * factor;
                }

                // Check for pulses at waveform end
                if (iChan <= 11 && h_wf.GetBinContent(ADCSIZE) > 100) {
                    pulse_at_end_count++;
                    if (pulse_at_end_count >= 10) pulse_at_end = true;
                }

                h_wf.Reset();
            }

            // Aggregate pulse properties
            p.start += mostFrequent(all_chan_start);
            p.end += mostFrequent(all_chan_end);
            p.energy = std::accumulate(all_chan_energy.begin(), all_chan_energy.end(), 0.0); // p.e.
            p.peak = std::accumulate(all_chan_peak.begin(), all_chan_peak.end(), 0.0); // p.e.
            p.side_vp_energy = std::accumulate(side_vp_energy.begin(), side_vp_energy.end(), 0.0); // ADC
            p.top_vp_energy = std::accumulate(top_vp_energy.begin(), top_vp_energy.end(), 0.0); // ADC
            p.all_vp_energy = p.side_vp_energy + p.top_vp_energy;

            // Check timing consistency
            for (const auto& start : all_chan_start) {
                if (fabs(start - mostFrequent(all_chan_start)) < 10 * 16.0 / 1000.0) { // Convert ns to µs
                    chan_starts_no_outliers.push_back(start);
                }
            }
            p.single = (variance(chan_starts_no_outliers) < 5 * 16.0 / 1000.0); // Convert ns to µs

            // Muon detection
            bool veto_hit = false;
            for (size_t i = 0; i < SIDE_VP_THRESHOLDS.size(); i++) {
                if (veto_energies[i] > SIDE_VP_THRESHOLDS[i]) {
                    veto_hit = true;
                    break;
                }
            }
            if (!veto_hit && p.top_vp_energy > TOP_VP_THRESHOLD) veto_hit = true;

            if ((p.energy > MUON_ENERGY_THRESHOLD && veto_hit) ||
                (pulse_at_end && p.energy > MUON_ENERGY_THRESHOLD / 2 && veto_hit)) {
                p.is_muon = true;
                last_muon_time = p.start; // Store in µs
                num_muons++;
                muon_candidates.emplace_back(p.start, p.energy); // Store muon candidate (µs)
                h_side_vp_muon->Fill(p.side_vp_energy);
                h_top_vp_muon->Fill(p.top_vp_energy);
            }

            // Michel electron detection
            double dt = p.start - last_muon_time; // dt in µs
            bool veto_low = true;
            for (size_t i = 0; i < SIDE_VP_THRESHOLDS.size(); i++) {
                if (veto_energies[i] > SIDE_VP_THRESHOLDS[i]) {
                    veto_low = false;
                    break;
                }
            }
            if (veto_energies[8] > TOP_VP_THRESHOLD || veto_energies[9] > TOP_VP_THRESHOLD) {
                veto_low = false;
            }

            if (p.energy >= MICHEL_ENERGY_MIN && p.energy <= MICHEL_ENERGY_MAX &&
                dt >= MICHEL_DT_MIN && dt <= MICHEL_DT_MAX && p.number >= 8 && veto_low &&
                p.trigger != 0 && p.trigger != 4 && p.trigger != 8 && p.trigger != 16) {
                p.is_michel = true;
                num_michels++;
                michel_muon_times.insert(last_muon_time); // Store muon time in µs
                h_michel_energy->Fill(p.energy);
                h_dt_michel->Fill(dt); // Fill in µs
                h_energy_vs_dt->Fill(dt, p.energy); // Fill in µs
            }

            p.last_muon_time = last_muon_time;

            // Fill tree
            br_eventID = eventID;
            br_start = p.start;
            br_energy = p.energy;
            br_peak = p.peak;
            br_number = p.number;
            br_side_vp_energy = p.side_vp_energy;
            br_top_vp_energy = p.top_vp_energy;
            br_all_vp_energy = p.all_vp_energy;
            br_last_muon_time = p.last_muon_time;
            br_dt_michel = dt;
            br_is_muon = p.is_muon;
            br_is_michel = p.is_michel;
            br_single = p.single;
            br_beam = p.beam;
            br_trigger = p.trigger;
            eventTree->Fill();
        }

        // Second pass: Fill h_muon_energy for muons associated with Michel electrons
        for (const auto& muon : muon_candidates) {
            if (michel_muon_times.find(muon.first) != michel_muon_times.end()) {
                h_muon_energy->Fill(muon.second);
            }
        }

        // Save output
        eventTree->Write();
        h_muon_energy->Write();
        h_michel_energy->Write();
        h_dt_michel->Write();
        h_energy_vs_dt->Write();
        h_side_vp_muon->Write();
        h_top_vp_muon->Write();
        fileOut->Write();
        fileOut->Close();
        f->Close();

        // Write stats
        statsFile << "File " << inputFileName << " Statistics:\n";
        statsFile << "Total Events: " << num_events << "\n";
        statsFile << "Muons Detected: " << num_muons << "\n";
        statsFile << "Michel Electrons Detected: " << num_michels << "\n";
        statsFile << "------------------------\n";

        num_events = 0;
        num_muons = 0;
        num_michels = 0;
    }

    // Check histogram contents
    cout << "Histogram contents before plotting:\n";
    cout << "  Muon Energy: " << h_muon_energy->GetEntries() << " entries\n";
    cout << "  Michel Energy: " << h_michel_energy->GetEntries() << " entries\n";
    cout << "  Michel dt: " << h_dt_michel->GetEntries() << " entries\n";
    cout << "  Energy vs dt: " << h_energy_vs_dt->GetEntries() << " entries\n";
    cout << "  Side Veto Muon: " << h_side_vp_muon->GetEntries() << " entries\n";
    cout << "  Top Veto Muon: " << h_top_vp_muon->GetEntries() << " entries\n";

    // Debug: Print h_dt_michel bin contents in fit range
    double fit_range_entries = h_dt_michel->Integral(h_dt_michel->FindBin(FIT_MIN), h_dt_michel->FindBin(FIT_MAX));
    cout << "h_dt_michel entries in fit range (1.1-16 µs): " << fit_range_entries << endl;
    cout << "h_dt_michel bin contents (1.1-16 µs):\n";
    int bin_min = h_dt_michel->FindBin(FIT_MIN);
    int bin_max = h_dt_michel->FindBin(FIT_MAX);
    for (int i = 1; i <= h_dt_michel->GetNbinsX(); i++) {
        double bin_center = h_dt_michel->GetBinCenter(i); // µs
        double content = h_dt_michel->GetBinContent(i);
        if (content > 0) {
            cout << Form("  Bin %d (%.2f µs): %.1f counts\n", i, bin_center, content);
        }
    }

    // Generate analysis plots
    TCanvas *c = new TCanvas("c", "Analysis Plots", 1200, 800);
    gStyle->SetOptStat(1111); // Show stats box
    gStyle->SetOptFit(1111);  // Include fit parameters in stats box

    // Muon Energy (only for muons with Michel electrons)
    c->Clear();
    h_muon_energy->SetLineColor(kBlue);
    h_muon_energy->Draw();
    c->Update();
    string plotName = OUTPUT_DIR + "/Muon_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Michel Energy
    c->Clear();
    h_michel_energy->SetLineColor(kRed);
    h_michel_energy->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Michel dt with exponential fit
    c->Clear();
    h_dt_michel->SetMarkerStyle(20);
    h_dt_michel->SetMarkerColor(kBlack+2);
    h_dt_michel->SetLineColor(kBlack+2);
    h_dt_michel->GetXaxis()->SetTitle("dt (#mus)");
    h_dt_michel->Draw("PE");

    if (h_dt_michel->GetEntries() > 5) { // Relaxed entry requirement
        // Estimate initial parameters
        double integral = h_dt_michel->Integral(h_dt_michel->FindBin(FIT_MIN), h_dt_michel->FindBin(FIT_MAX));
        double bin_width = h_dt_michel->GetBinWidth(1); // µs
        double N0_init = integral * bin_width / (FIT_MAX - FIT_MIN); // Approximate counts per µs
        // Use minimum bin content in 14-16 µs for C_init
        double C_init = 0;
        int bin_14 = h_dt_michel->FindBin(14.0);
        int bin_16 = h_dt_michel->FindBin(16.0);
        double min_content = 1e9;
        for (int i = bin_14; i <= bin_16; i++) {
            double content = h_dt_michel->GetBinContent(i);
            if (content > 0 && content < min_content) min_content = content;
        }
        if (min_content < 1e9) C_init = min_content;
        else C_init = 0.1; // Fallback if no counts

        TF1 *expFit = new TF1("expFit", ExpFit, FIT_MIN, FIT_MAX, 3);
        expFit->SetParameters(N0_init, 2.2, C_init); // τ in µs
        expFit->SetParLimits(0, 0, N0_init * 100); // N0, relaxed upper limit
        expFit->SetParLimits(1, 0.1, 20.0); // tau (µs)
        expFit->SetParLimits(2, -C_init * 10, C_init * 10); // C, allow negative
        expFit->SetParNames("N_{0}", "#tau", "C");
        expFit->SetNpx(1000); // Smoother fit evaluation

        // Perform fit with simpler options
        int fitStatus = h_dt_michel->Fit(expFit, "RE", "", FIT_MIN, FIT_MAX);
        expFit->SetLineColor(2); // Explicitly kRed
        expFit->SetLineWidth(3);
        expFit->Draw("same");
        gPad->Update();
        cout << "Fit line drawn with color kRed (2), width 3" << endl;

        // Customize stats box
        TPaveStats *stats = (TPaveStats*)h_dt_michel->FindObject("stats");
        if (!stats) {
            cout << "Stats box not found, creating new TPaveStats" << endl;
            stats = new TPaveStats(0.60, 0.60, 0.90, 0.90, "brNDC");
            stats->SetName("stats");
            h_dt_michel->GetListOfFunctions()->Add(stats);
        } else {
            cout << "Stats box found, updating content" << endl;
        }
        stats->SetTextColor(2); // kRed
        stats->SetX1NDC(0.60);
        stats->SetX2NDC(0.90);
        stats->SetY1NDC(0.60);
        stats->SetY2NDC(0.90);
        stats->Clear();
        stats->AddText(Form("#tau = %.4f #pm %.4f #mus", expFit->GetParameter(1), expFit->GetParError(1)));
        stats->AddText(Form("#chi^{2}/NDF = %.4f", expFit->GetChisquare() / expFit->GetNDF()));
        stats->AddText(Form("N_{0} = %.1f #pm %.1f", expFit->GetParameter(0), expFit->GetParError(0)));
        stats->AddText(Form("C = %.1f #pm %.1f", expFit->GetParameter(2), expFit->GetParError(2)));
        stats->Draw();
        gPad->Update();

        // Print fit results in µs
        double N0 = expFit->GetParameter(0);
        double N0_err = expFit->GetParError(0);
        double tau = expFit->GetParameter(1);
        double tau_err = expFit->GetParError(1);
        double C = expFit->GetParameter(2);
        double C_err = expFit->GetParError(2);
        double chi2 = expFit->GetChisquare();
        int ndf = expFit->GetNDF();
        double chi2_ndf = ndf > 0 ? chi2 / ndf : 0;

        cout << "Exponential Fit Results (Michel dt, 1.1-16 µs):\n";
        cout << Form("Fit Status: %d (0 = success)", fitStatus) << endl;
        cout << Form("N₀ = %.1f ± %.1f", N0, N0_err) << endl;
        cout << Form("τ = %.4f ± %.4f µs", tau, tau_err) << endl;
        cout << Form("C = %.1f ± %.1f", C, C_err) << endl;
        cout << Form("χ² = %.1f", chi2) << endl;
        cout << Form("NDF = %d", ndf) << endl;
        cout << Form("χ²/NDF = %.4f", chi2_ndf) << endl;
        cout << "----------------------------------------" << endl;

        if (fitStatus != 0) {
            cout << "Warning: Exponential fit failed for h_dt_michel (status = " << fitStatus << ")" << endl;
            cout << "Initial Parameters: N0 = " << N0_init << ", τ = 2.2 µs, C = " << C_init << endl;
            cout << "Fit results may be unreliable, but drawn for inspection." << endl;
            cout << "Check h_dt_michel bin contents above for data distribution." << endl;
        }
        delete expFit;
    } else {
        cout << "Warning: h_dt_michel has insufficient entries (" << h_dt_michel->GetEntries() << "), skipping exponential fit" << endl;
        cout << "Check Michel electron detection criteria (e.g., MICHEL_ENERGY_MIN, p.number, veto thresholds)." << endl;
    }

    c->Update();
    c->Modified();
    c->RedrawAxis(); // Ensure axes and fit are visible
    plotName = OUTPUT_DIR + "/Michel_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Energy vs dt
    c->Clear();
    h_energy_vs_dt->GetXaxis()->SetTitle("dt (#mus)");
    h_energy_vs_dt->Draw("COLZ");
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy_vs_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Side Veto Muon
    c->Clear();
    h_side_vp_muon->SetLineColor(kMagenta);
    h_side_vp_muon->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Side_Veto_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Top Veto Muon
    c->Clear();
    h_top_vp_muon->SetLineColor(kCyan);
    h_top_vp_muon->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Top_Veto_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    statsFile.close();

    // Clean up
    delete h_muon_energy;
    delete h_michel_energy;
    delete h_dt_michel;
    delete h_energy_vs_dt;
    delete h_side_vp_muon;
    delete h_top_vp_muon;
    delete c;

    cout << "Analysis complete. Results saved in " << OUTPUT_DIR << "/ (ROOT files, " << OUTPUT_STATS_NAME << ", and *.png)" << endl;
    return 0;
}
