//Muon and Michel Electron Analysis Code
//This code comprehensively analyses cosmic muon events and their subsequent Michel electron decay in a particle detector. The analysis includes:
//PMT calibration using LED data Muon event selection with SiPM shower rejection
//Michel electron identification in subsequent events Measurement of muon lifetime from time  differences Energy spectrum analysis of Michel electrons#include <iostream>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TPaveText.h>
#include <TChain.h>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <sys/stat.h>
#include <ctime>
#include <cstdlib>
#include <TLine.h>
#include <TLatex.h>

using namespace std;

// Constants
const int N_PMTS = 12;
const int N_SIPMS = 10;
const int PMT_CHANNEL_MAP[N_PMTS] = {0,10,7,2,6,3,8,9,11,4,5,1};
const int SIPM_CHANNEL_MAP[N_SIPMS] = {12,13,14,15,16,17,18,19,20,21};
const double SIPM_THRESHOLDS[N_SIPMS] = {800,800,1100,1200,550,600,650,450,600,650};
const double PMT_THRESHOLDS[N_PMTS] = {4800,6000,5000,6000,6000,4700,4500,3000,2000,5000,4500,4800};

// SPE fitting function (4-Gaussian model)
Double_t SPEfit(Double_t *x, Double_t *par) {
    Double_t term1 = par[0] * exp(-0.5 * pow((x[0]-par[1])/par[2], 2));
    Double_t term2 = par[3] * exp(-0.5 * pow((x[0]-par[4])/par[5], 2));
    Double_t term3 = par[6] * exp(-0.5 * pow((x[0]-sqrt(2)*par[4])/sqrt(2*pow(par[5],2)-pow(par[2],2)), 2));
    Double_t term4 = par[7] * exp(-0.5 * pow((x[0]-sqrt(3)*par[4])/sqrt(3*pow(par[5],2)-2*pow(par[2],2)), 2));
    return term1 + term2 + term3 + term4;
}

// Muon decay exponential function
Double_t DecayFit(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0]/par[1]);
}

// Calculate mean and RMS of a vector
void CalculateMeanAndRMS(const vector<Double_t> &data, Double_t &mean, Double_t &rms) {
    mean = 0.0;
    for (const auto &value : data) mean += value;
    mean /= data.size();
    
    rms = 0.0;
    for (const auto &value : data) rms += pow(value - mean, 2);
    rms = sqrt(rms / data.size());
}

// Generate unique output directory name
string generateOutputDirName() {
    time_t now = time(0);
    tm *ltm = localtime(&now);
    int randomNum = rand() % 10000;
    
    return "MichelAnalysis_" + to_string(1900 + ltm->tm_year) + 
           to_string(1 + ltm->tm_mon) + to_string(ltm->tm_mday) + "_" +
           to_string(ltm->tm_hour) + to_string(ltm->tm_min) + to_string(ltm->tm_sec) +
           "_" + to_string(randomNum);
}

// PMT calibration function
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

    TH1F *histArea[N_PMTS];
    Long64_t nLEDFlashes[N_PMTS] = {0};
    
    // Initialize histograms
    for (int i=0; i<N_PMTS; i++) {
        histArea[i] = new TH1F(Form("PMT%d_Area",i+1), 
                             Form("PMT %d;ADC Counts;Events",i+1), 150, -50, 400);
    }

    // Set branch addresses
    Int_t triggerBits;
    Double_t area[23];
    calibTree->SetBranchAddress("triggerBits", &triggerBits);
    calibTree->SetBranchAddress("area", area);

    // Process calibration events
    Long64_t nEntries = calibTree->GetEntries();
    cout << "Processing " << nEntries << " calibration events..." << endl;
    
    for (Long64_t entry=0; entry<nEntries; entry++) {
        calibTree->GetEntry(entry);
        if (triggerBits != 16) continue;  // Only use LED events
        
        for (int pmt=0; pmt<N_PMTS; pmt++) {
            histArea[pmt]->Fill(area[PMT_CHANNEL_MAP[pmt]]);
            nLEDFlashes[pmt]++;
        }
    }

    // Perform SPE fits for each PMT
    for (int i=0; i<N_PMTS; i++) {
        if (histArea[i]->GetEntries() < 1000) {
            cerr << "Warning: Insufficient data for PMT " << i+1 << endl;
            mu1[i] = 0;
            mu1_err[i] = 0;
            continue;
        }

        TF1 *fitFunc = new TF1("fitFunc", SPEfit, -50, 400, 8);
        Double_t histMean = histArea[i]->GetMean();
        Double_t histRMS = histArea[i]->GetRMS();

        // Set initial parameters for SPE fit
        fitFunc->SetParameters(1000, histMean-histRMS, histRMS/2,
                             1000, histMean, histRMS,
                             500, 200);
        
        histArea[i]->Fit(fitFunc, "Q0", "", -50, 400);
        mu1[i] = fitFunc->GetParameter(4); // 1PE mean position
        Double_t sigma_mu1 = fitFunc->GetParError(4);
        Double_t sigma1 = fitFunc->GetParameter(5);
        
        // Calculate calibration error
        mu1_err[i] = sqrt(pow(sigma_mu1, 2) + pow(sigma1/sqrt(nLEDFlashes[i]), 2));
        
        delete fitFunc;
        delete histArea[i];
    }

    // Print calibration results
    cout << "\nPMT Calibration Results (1PE peak positions):\n";
    cout << "PMT#  HardwareCh  mu1 [ADC]  Error [ADC]  N_events\n";
    cout << "--------------------------------------------------\n";
    for (int i=0; i<N_PMTS; i++) {
        printf("PMT%02d     %2d       %6.2f      %5.2f      %6lld\n", 
              i+1, PMT_CHANNEL_MAP[i], mu1[i], mu1_err[i], nLEDFlashes[i]);
    }
    cout << endl;

    calibFile->Close();
}

// Main event selection function
bool passesMainEventSelection(const Double_t *pulseH, const Double_t *baselineRMS, 
                            const Double_t *area, const Int_t *peakPosition,
                            const Double_t *mu1) {
    // Condition A: Pulse Height > 2 p.e. for at least 3 PMTs
    int countAbove2PE = 0;
    for (int pmt=0; pmt<N_PMTS; pmt++) {
        if (pulseH[PMT_CHANNEL_MAP[pmt]] > 2 * mu1[pmt]) {
            countAbove2PE++;
        }
    }

    if (countAbove2PE >= 3) {
        vector<Double_t> peakPositions;
        for (int pmt=0; pmt<N_PMTS; pmt++) {
            peakPositions.push_back(peakPosition[PMT_CHANNEL_MAP[pmt]]);
        }
        Double_t dummyMean, currentRMS;
        CalculateMeanAndRMS(peakPositions, dummyMean, currentRMS);
        if (currentRMS < 2.5) return true;
    } 
    else {
        // Condition B: Pulse Height > 3 * baseline RMS and area/height > 1.2
        int countConditionB = 0;
        for (int pmt=0; pmt<N_PMTS; pmt++) {
            int ch = PMT_CHANNEL_MAP[pmt];
            if (pulseH[ch] > 3 * baselineRMS[ch] && (area[ch] / pulseH[ch]) > 1.2) {
                countConditionB++;
            }
        }

        if (countConditionB >= 3) {
            vector<Double_t> peakPositions;
            for (int pmt=0; pmt<N_PMTS; pmt++) {
                peakPositions.push_back(peakPosition[PMT_CHANNEL_MAP[pmt]]);
            }
            Double_t dummyMean, currentRMS;
            CalculateMeanAndRMS(peakPositions, dummyMean, currentRMS);
            if (currentRMS < 2.5) return true;
        }
    }
    return false;
}

// Main analysis function
void analyzeMuonMichel(TChain *analysisChain, const Double_t *mu1, const string &outputDir) {
    gErrorIgnoreLevel = kError;  // Suppress ROOT info messages

    // Variables to read from the tree
    Double_t area[23], pulseH[23], baselineRMS[23];
    Int_t triggerBits, peakPosition[23];
    Long64_t nsTime;

    // Set branch addresses
    analysisChain->SetBranchAddress("triggerBits", &triggerBits);
    analysisChain->SetBranchAddress("area", area);
    analysisChain->SetBranchAddress("pulseH", pulseH);
    analysisChain->SetBranchAddress("baselineRMS", baselineRMS);
    analysisChain->SetBranchAddress("peakPosition", peakPosition);
    analysisChain->SetBranchAddress("nsTime", &nsTime);

    // Create output directory
    mkdir(outputDir.c_str(), 0777);

    // Create histograms with adjusted ranges
    TH1F *histDeltaT = new TH1F("DeltaT", 
        "Muon-Michel Time Difference;Time Difference (#mus);Counts/0.1 #mus", 
        135, 1.5, 15);  // 135 bins from 1.5-15 μs (0.1 μs/bin)
        
    TH1F *histMichelSpectrum = new TH1F("MichelSpectrum", 
        "Michel Electron Energy;Photoelectrons;Events/10 p.e.", 
        100, 0, 1000);
        
    TH1F *histMuonPMTHits = new TH1F("MuonPMTHits", 
        "PMT Multiplicity;Number of PMTs hit;Events", 
        12, 0, 12);
        
    TH1F *histSiPMMultiplicity = new TH1F("SiPMMultiplicity", 
        "SiPM Multiplicity;Number of SiPMs hit;Events", 
        10, 0, 10);
        
    TH1F *histTriggerBits = new TH1F("TriggerBits", 
        "Trigger Bits Distribution;Trigger Bits;Events", 
        64, 0, 64);

    int muonCount = 0, michelCount = 0;
    int totalTrigger34Events = 0;

    // Event loop
    Long64_t nEntries = analysisChain->GetEntries();
    cout << "Analyzing " << nEntries << " events..." << endl;

    for(Long64_t entry=0; entry<nEntries; entry++) {
        analysisChain->GetEntry(entry);
        
        // Monitor trigger bits
        histTriggerBits->Fill(triggerBits);
        
        // Only process events with triggerBits == 34 or 2
        if (triggerBits != 34 && triggerBits != 2) continue;
        totalTrigger34Events++;
        
        // Apply main event selection
        if (!passesMainEventSelection(pulseH, baselineRMS, area, peakPosition, mu1)) {
            continue;
        }

        // Count SiPM hits above threshold
        int sipmHitCount = 0;
        for (int i = 0; i < N_SIPMS; i++) {
            int ch = SIPM_CHANNEL_MAP[i];
            if (area[ch] >= SIPM_THRESHOLDS[i]) {
                sipmHitCount++;
            }
        }
        
        // Fill SiPM multiplicity histogram
        histSiPMMultiplicity->Fill(sipmHitCount);
        
        // Apply shower rejection cut (multiplicity ≤ 2)
        if (sipmHitCount > 2) continue;

        // PMT multiplicity calculation
        int pmtHitCount = 0;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            if (area[PMT_CHANNEL_MAP[pmt]] >= PMT_THRESHOLDS[pmt]) {
                pmtHitCount++;
            }
        }
        
        // Fill PMT multiplicity histogram for all events passing previous cuts
        histMuonPMTHits->Fill(pmtHitCount);

        // Valid muon candidate found (now without PMT multiplicity cut)
        muonCount++;
        Long64_t muonTime = nsTime;
        
        // Michel electron search (1-10 μs window)
        for(Long64_t nextEntry = entry + 1; nextEntry < nEntries; nextEntry++) {
            analysisChain->GetEntry(nextEntry);
            
            if (triggerBits != 34 && triggerBits != 2) continue;
            
            double deltaT = (nsTime - muonTime) * 1e-3; // ns to μs
            if (deltaT > 10) break;
            if (deltaT < 1) continue;

            if (!passesMainEventSelection(pulseH, baselineRMS, area, peakPosition, mu1)) {
                continue;
            }

            // Michel condition: ≥11 PMTs with ≥2 PE
            int michelPMTCount = 0;
            double michelEnergy = 0.0;
            for (int pmt = 0; pmt < N_PMTS; pmt++) {
                double pe = area[PMT_CHANNEL_MAP[pmt]] / mu1[pmt];
                if (pe >= 2) {
                    michelPMTCount++;
                    michelEnergy += pe;
                }
            }

            if (michelPMTCount >= 11) {
                michelCount++;
                histDeltaT->Fill(deltaT);
                histMichelSpectrum->Fill(michelEnergy);
                break; // Take first valid Michel
            }
        }
    }

    // Analysis summary
    cout << "\nANALYSIS SUMMARY:" << endl;
    cout << "Total events processed: " << nEntries << endl;
    cout << "TriggerBits==34 events: " << totalTrigger34Events << endl;
    cout << "Muon candidates: " << muonCount << endl;
    cout << "Michel electrons: " << michelCount << endl;
    cout << "Michel fraction: " << (muonCount > 0 ? 100.0*michelCount/muonCount : 0) << "%" << endl;

    // Plotting function
    auto plotHistogram = [&outputDir](TH1F* hist, const string& name, 
                                    const string& title, bool addCutLine = false, 
                                    double cutValue = 0) {
        TCanvas *c = new TCanvas(name.c_str(), title.c_str(), 1200, 800);
        hist->Draw();
        
        c->SaveAs((outputDir + "/" + name + ".png").c_str());
        delete c;
    };

    // 1. Time difference with lifetime fit
    TCanvas *c1 = new TCanvas("c1", "Time Difference", 1200, 800);
    TF1 *decayFit = new TF1("decayFit", DecayFit, 1.5, 15, 2);
    decayFit->SetParameters(histDeltaT->GetMaximum(), 2.2);
    decayFit->SetParLimits(0, 0, histDeltaT->GetMaximum() * 2);
    decayFit->SetParLimits(1, 0.1, 15);
    
    histDeltaT->Fit(decayFit, "L", "", 1.5, 15);
    
    TPaveText *pt = new TPaveText(0.55, 0.65, 0.85, 0.8, "NDC");
    pt->SetFillColor(0);
    pt->SetBorderSize(1);
    pt->SetTextAlign(12);
    pt->SetTextSize(0.035);
    pt->AddText(Form("#tau = %.2f #pm %.2f #mus", 
                    decayFit->GetParameter(1), 
                    decayFit->GetParError(1)));
    
    histDeltaT->GetListOfFunctions()->Add(pt);
    histDeltaT->GetXaxis()->SetRangeUser(1.5, 10); // Display 1.5-10 μs
    histDeltaT->Draw();
    c1->SaveAs((outputDir + "/time_difference.png").c_str());

    // Other plots
    plotHistogram(histMichelSpectrum, "michel_spectrum", "Michel Electron Energy Spectrum");
    plotHistogram(histMuonPMTHits, "muon_pmt_hits", "PMT Multiplicity");
    plotHistogram(histSiPMMultiplicity, "sipm_multiplicity", 
                 "SiPM Multiplicity Distribution", true, 2.5);
    plotHistogram(histTriggerBits, "trigger_bits", "Trigger Bits Distribution");

    // Save all histograms to ROOT file
    TFile *outFile = new TFile((outputDir + "/results.root").c_str(), "RECREATE");
    histDeltaT->Write();
    histMichelSpectrum->Write();
    histMuonPMTHits->Write();
    histSiPMMultiplicity->Write();
    histTriggerBits->Write();
    outFile->Close();

    // Cleanup
    delete histDeltaT; delete histMichelSpectrum;
    delete histMuonPMTHits; delete histSiPMMultiplicity;
    delete histTriggerBits;
    delete c1; delete outFile;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file1.root> [input_file2.root ...]" << endl;
        cerr << "Note: First file used for calibration (triggerBits=16), others for analysis" << endl;
        return 1;
    }

    string outputDir = generateOutputDirName();
    cout << "Creating output directory: " << outputDir << endl;
    mkdir(outputDir.c_str(), 0777);

    // Perform PMT calibration
    Double_t mu1[N_PMTS], mu1_err[N_PMTS];
    performCalibration(argv[1], mu1, mu1_err);

    // Set up analysis chain
    TChain *analysisChain = new TChain("tree");
    for (int i = 1; i < argc; i++) {
        analysisChain->Add(argv[i]);
        cout << "Added file: " << argv[i] << endl;
    }

    // Run analysis
    analyzeMuonMichel(analysisChain, mu1, outputDir);
    delete analysisChain;

    cout << "Analysis complete. Results in: " << outputDir << endl;
    return 0;
}
