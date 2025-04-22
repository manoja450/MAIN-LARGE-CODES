// Include necessary ROOT and standard library headers
#include <iostream>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TPaveText.h>
#include <TChain.h>
#include <TLegend.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TStyle.h>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <sys/stat.h>
#include <ctime>
#include <cstdlib>
#include <TLine.h>
#include <TLatex.h>
#include <TGaxis.h>
#include <algorithm>

using namespace std;

// ======================================================================
// Constants for detector configuration
// ======================================================================
const int N_PMTS = 12;                 // Number of PMTs in the detector
const int N_SIPMS = 10;                // Number of SiPMs in the detector

// Hardware channel mapping for PMTs
const int PMT_CHANNEL_MAP[N_PMTS] = {0,10,7,2,6,3,8,9,11,4,5,1};

// Hardware channel mapping for SiPMs
const int SIPM_CHANNEL_MAP[N_SIPMS] = {12,13,14,15,16,17,18,19,20,21};

// Thresholds for SiPMs in ADC counts
const double SIPM_THRESHOLDS[N_SIPMS] = {800,800,1000,1000,550,600,650,450,600,650};

// Thresholds for PMTs in ADC counts
const double PMT_THRESHOLDS[N_PMTS] = {4800,6000,5000,6000,6000,4707,4500,3000,2000,5000,4500,4800};

// Cross-talk probability between channels
const double CROSS_TALK_PROBABILITY = 0.05;

// Afterpulsing probability in PMTs
const double AFTERPULSE_PROBABILITY = 0.03;


// ======================================================================
// Mathematical Functions
// ======================================================================

// Single Photoelectron (SPE) fitting function
// Uses four Gaussian terms to model the SPE response:
// - Pedestal
// - 1PE peak
// - 2PE peak
// - 3PE peak
Double_t SPEfit(Double_t *x, Double_t *par) {
    // Pedestal peak
    Double_t term1 = par[0] * exp(-0.5 * pow((x[0]-par[1])/par[2], 2));
    
    // 1PE peak
    Double_t term2 = par[3] * exp(-0.5 * pow((x[0]-par[4])/par[5], 2));
    
    // 2PE peak (with proper width scaling)
    Double_t term3 = par[6] * exp(-0.5 * pow((x[0]-sqrt(2)*par[4])/sqrt(2*pow(par[5],2)-pow(par[2],2)), 2));
    
    // 3PE peak (with proper width scaling)
    Double_t term4 = par[7] * exp(-0.5 * pow((x[0]-sqrt(3)*par[4])/sqrt(3*pow(par[5],2)-2*pow(par[2],2)), 2));
    
    return term1 + term2 + term3 + term4;
}

// Muon decay exponential fitting function
// Simple exponential decay with two parameters:
// - par[0]: Normalization constant
// - par[1]: Lifetime (tau)
Double_t DecayFit(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0]/par[1]);
}


// ======================================================================
// Utility Functions
// ======================================================================

// Calculate mean and RMS of a data vector
void CalculateMeanAndRMS(const vector<Double_t> &data, Double_t &mean, Double_t &rms) {
    // Calculate mean
    mean = 0.0;
    for (const auto &value : data) mean += value;
    mean /= data.size();
    
    // Calculate RMS
    rms = 0.0;
    for (const auto &value : data) rms += pow(value - mean, 2);
    rms = sqrt(rms / data.size());
}

// Generate unique output directory name with timestamp
string generateOutputDirName() {
    time_t now = time(0);
    tm *ltm = localtime(&now);
    int randomNum = rand() % 10000;
    return "MichelAnalysis_" + to_string(1900 + ltm->tm_year) +
           to_string(1 + ltm->tm_mon) + to_string(ltm->tm_mday) + "_" +
           to_string(ltm->tm_hour) + to_string(ltm->tm_min) + to_string(ltm->tm_sec) +
           "_" + to_string(randomNum);
}


// ======================================================================
// Calibration Functions
// ======================================================================

// PMT calibration using LED data
// Fits single photoelectron spectra for each PMT
void performCalibration(const string &calibFileName, Double_t *mu1, Double_t *mu1_err) {
    // Open calibration file
    TFile *calibFile = TFile::Open(calibFileName.c_str());
    if (!calibFile || calibFile->IsZombie()) {
        cerr << "Error opening calibration file: " << calibFileName << endl;
        exit(1);
    }

    // Access calibration tree
    TTree *calibTree = (TTree*)calibFile->Get("tree");
    if (!calibTree) {
        cerr << "Error accessing tree in calibration file" << endl;
        calibFile->Close();
        exit(1);
    }

    // Create histograms for each PMT
    TH1F *histArea[N_PMTS];
    Long64_t nLEDFlashes[N_PMTS] = {0};
    for (int i = 0; i < N_PMTS; i++) {
        histArea[i] = new TH1F(Form("PMT%d_Area", i+1),
                               Form("PMT %d;ADC Counts;Events", i+1), 150, -50, 400);
    }

    // Set up branch addresses
    Int_t triggerBits;
    Double_t area[23];
    calibTree->SetBranchAddress("triggerBits", &triggerBits);
    calibTree->SetBranchAddress("area", area);

    // Process calibration events
    Long64_t nEntries = calibTree->GetEntries();
    cout << "Processing " << nEntries << " calibration events..." << endl;

    for (Long64_t entry = 0; entry < nEntries; entry++) {
        calibTree->GetEntry(entry);
        
        // Only use LED trigger events (triggerBits == 16)
        if (triggerBits != 16) continue;
        
        // Fill histograms for each PMT
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            histArea[pmt]->Fill(area[PMT_CHANNEL_MAP[pmt]]);
            nLEDFlashes[pmt]++;
        }
    }

    // Fit each PMT's SPE spectrum
    for (int i = 0; i < N_PMTS; i++) {
        // Skip PMTs with insufficient data
        if (histArea[i]->GetEntries() < 1000) {
            cerr << "Warning: Insufficient data for PMT " << i+1 << endl;
            mu1[i] = 0;
            mu1_err[i] = 0;
            continue;
        }

        // Create and configure fit function
        TF1 *fitFunc = new TF1("fitFunc", SPEfit, -50, 400, 8);
        Double_t histMean = histArea[i]->GetMean();
        Double_t histRMS = histArea[i]->GetRMS();
        
        // Set initial parameters
        fitFunc->SetParameters(1000, histMean - histRMS, histRMS/2,
                              1000, histMean, histRMS,
                              500, 200);
        
        // Perform fit
        histArea[i]->Fit(fitFunc, "Q0", "", -50, 400);
        
        // Extract results
        mu1[i] = fitFunc->GetParameter(4);  // 1PE peak position
        Double_t sigma_mu1 = fitFunc->GetParError(4);
        Double_t sigma1 = fitFunc->GetParameter(5);
        
        // Calculate combined uncertainty
        mu1_err[i] = sqrt(pow(sigma_mu1, 2) + pow(sigma1/sqrt(nLEDFlashes[i]), 2));
        
        // Clean up
        delete fitFunc;
        delete histArea[i];
    }

    // Print calibration results
    cout << "\nPMT Calibration Results (1PE peak positions):\n";
    cout << "PMT#  HardwareCh  mu1 [ADC]  Error [ADC]  N_events\n";
    cout << "--------------------------------------------------\n";
    for (int i = 0; i < N_PMTS; i++) {
        printf("PMT%02d     %2d       %6.2f      %5.2f      %6lld\n", 
               i+1, PMT_CHANNEL_MAP[i], mu1[i], mu1_err[i], nLEDFlashes[i]);
    }
    cout << endl;
    
    // Close calibration file
    calibFile->Close();
}


// ======================================================================
// Event Selection Criteria
// ======================================================================

// Determine if event passes main selection criteria
bool passesMainEventSelection(const Double_t *pulseH, const Double_t *baselineRMS,
                             const Double_t *area, const Int_t *peakPosition,
                             const Double_t *mu1) {
    // First condition: At least 3 PMTs with >2PE signal
    int countAbove2PE = 0;
    for (int pmt = 0; pmt < N_PMTS; pmt++) {
        if (pulseH[PMT_CHANNEL_MAP[pmt]] > 2 * mu1[pmt])
            countAbove2PE++;
    }

    if (countAbove2PE >= 3) {
        // Check timing consistency
        vector<Double_t> peakPositions;
        for (int pmt = 0; pmt < N_PMTS; pmt++)
            peakPositions.push_back(peakPosition[PMT_CHANNEL_MAP[pmt]]);
        
        Double_t dummyMean, currentRMS;
        CalculateMeanAndRMS(peakPositions, dummyMean, currentRMS);
        
        if (currentRMS < 2.5) return true;
    }
    else {
        // Second condition: Alternative selection
        int countConditionB = 0;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            int ch = PMT_CHANNEL_MAP[pmt];
            if (pulseH[ch] > 3 * baselineRMS[ch] && (area[ch] / pulseH[ch]) > 1.2)
                countConditionB++;
        }
        
        if (countConditionB >= 3) {
            // Check timing consistency
            vector<Double_t> peakPositions;
            for (int pmt = 0; pmt < N_PMTS; pmt++)
                peakPositions.push_back(peakPosition[PMT_CHANNEL_MAP[pmt]]);
            
            Double_t dummyMean, currentRMS;
            CalculateMeanAndRMS(peakPositions, dummyMean, currentRMS);
            
            if (currentRMS < 2.5) return true;
        }
    }
    
    return false;
}


// ======================================================================
// Main Analysis Function
// ======================================================================

void analyzeMuonMichel(TChain *analysisChain, const Double_t *mu1, const string &outputDir) {
    // Suppress ROOT info messages
    gErrorIgnoreLevel = kError;
    
    // ==================================================================
    // Set up tree branches
    // ==================================================================
    Double_t area[23], pulseH[23], baselineRMS[23];
    Int_t triggerBits, peakPosition[23];
    Long64_t nsTime;
    
    analysisChain->SetBranchAddress("triggerBits", &triggerBits);
    analysisChain->SetBranchAddress("area", area);
    analysisChain->SetBranchAddress("pulseH", pulseH);
    analysisChain->SetBranchAddress("baselineRMS", baselineRMS);
    analysisChain->SetBranchAddress("peakPosition", peakPosition);
    analysisChain->SetBranchAddress("nsTime", &nsTime);

    // Create output directory
    mkdir(outputDir.c_str(), 0777);

    // ==================================================================
    // Create histograms
    // ==================================================================
    
    // Time difference between muon and Michel electron
    TH1F *histDeltaT = new TH1F("DeltaT",
        "Muon-Michel Time Difference;Time to previous event(Muon)(#mus);Counts/0.1 #mus",
        100, 1.0, 10.0);
    
    // Michel electron energy spectrum
    TH1F *histMichelSpectrum = new TH1F("MichelSpectrum",
        "Michel Electron Energy;Photoelectrons;Counts/10 p.e.", 100, 0, 1000);
    
    // Accidental background spectrum
    TH1F *histMichelSpectrum_Accidental = new TH1F("MichelSpectrum_Accidental",
        "Michel Electron Energy (Accidental);Photoelectrons;Counts/10 p.e.", 100, 0, 1000);
    
    // PMT multiplicity for muons
    TH1F *histMuonPMTHits = new TH1F("MuonPMTHits",
        "PMT Multiplicity;Number of PMTs hit;Counts/bin", 12, 0, 12);
    
    // SiPM multiplicity
    TH1F *histSiPMMultiplicity = new TH1F("SiPMMultiplicity",
        "SiPM Multiplicity;Number of SiPMs hit;Counts/bin", 10, 0, 10);
    
    // Trigger bits distribution
    TH1F *histTriggerBits = new TH1F("TriggerBits",
        "Trigger Bits Distribution;Trigger Bits;Counts/bin", 36, 0, 36);
    
    // Muon energy spectrum
    TH1F *histMuonSpectrum = new TH1F("MuonSpectrum",
        "Muon Spectrum ;Photoelectrons;Events/50 p.e.",
        100, 500, 5000);

    // Accidental time spectrum
    TH1F *histAccidental = new TH1F("AccidentalDeltaT",
        "Accidental Time Spectrum;Time (#mus);Counts/0.1 #mus", 100, 10.0, 20.0);

    // ==================================================================
    // Set histogram styles
    // ==================================================================
    histDeltaT->GetXaxis()->SetTitleOffset(1.2);
    histDeltaT->GetYaxis()->SetTitleOffset(1.2);
    histDeltaT->GetYaxis()->SetTitleSize(0.04);
    
    histMichelSpectrum->GetXaxis()->SetTitleOffset(1.2);
    histMichelSpectrum->GetYaxis()->SetTitleOffset(1.2);
    histMichelSpectrum->GetYaxis()->SetTitleSize(0.04);
    
    histMichelSpectrum_Accidental->GetXaxis()->SetTitleOffset(1.2);
    histMichelSpectrum_Accidental->GetYaxis()->SetTitleOffset(1.2);
    histMichelSpectrum_Accidental->GetYaxis()->SetTitleSize(0.04);
    
    histMuonPMTHits->GetXaxis()->SetTitleOffset(1.2);
    histMuonPMTHits->GetYaxis()->SetTitleOffset(1.2);
    histMuonPMTHits->GetYaxis()->SetTitleSize(0.04);
    
    histSiPMMultiplicity->GetXaxis()->SetTitleOffset(1.2);
    histSiPMMultiplicity->GetYaxis()->SetTitleOffset(1.2);
    histSiPMMultiplicity->GetYaxis()->SetTitleSize(0.04);
    
    histTriggerBits->GetXaxis()->SetTitleOffset(1.2);
    histTriggerBits->GetYaxis()->SetTitleOffset(1.2);
    histTriggerBits->GetYaxis()->SetTitleSize(0.04);
    
    histMuonSpectrum->GetXaxis()->SetTitleOffset(1.2);
    histMuonSpectrum->GetYaxis()->SetTitleOffset(1.4);
    histMuonSpectrum->GetYaxis()->SetTitleSize(0.04);
    
    histAccidental->GetYaxis()->SetTitleOffset(1.2);
    histAccidental->GetYaxis()->SetTitleSize(0.04);

    // ==================================================================
    // Event processing
    // ==================================================================
    int muonCount = 0, michelCount = 0, totalTrigger34Events = 0;
    Long64_t lastBeamTime = -1000000;  
    Long64_t nEntries = analysisChain->GetEntries();
    cout << "Analyzing " << nEntries << " events..." << endl;

    for (Long64_t entry = 0; entry < nEntries; entry++) {
        analysisChain->GetEntry(entry);
        
        // Skip beam events and events too close to beam
        if (triggerBits == 1) {
            lastBeamTime = nsTime;
            continue;
        }
        if ((nsTime - lastBeamTime) < 10000) continue;
        
        // Fill trigger bits histogram
        histTriggerBits->Fill(triggerBits);
        
        // Only process specific trigger types
        if (triggerBits != 34 && triggerBits != 2 && triggerBits != 32) continue;
        totalTrigger34Events++;
        
        // Apply main event selection
        if (!passesMainEventSelection(pulseH, baselineRMS, area, peakPosition, mu1)) continue;
        
        // Count SiPM hits
        int sipmHitCount = 0;
        for (int i = 0; i < N_SIPMS; i++) {
            int ch = SIPM_CHANNEL_MAP[i];
            if (area[ch] >= SIPM_THRESHOLDS[i]) sipmHitCount++;
        }
        histSiPMMultiplicity->Fill(sipmHitCount);
        
        // Require 1-3 SiPM hits
        if (sipmHitCount < 1 || sipmHitCount > 3) continue;

        // Count PMT hits
        int pmtHitCount = 0;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            if (area[PMT_CHANNEL_MAP[pmt]] >= PMT_THRESHOLDS[pmt]) pmtHitCount++;
        }
        histMuonPMTHits->Fill(pmtHitCount);
        
        // Look for muon candidates (>10 PMT hits)
        if (pmtHitCount > 10) {
            // Calculate muon energy
            double sumPE = 0.0;
            for (int pmt = 0; pmt < N_PMTS; pmt++) {
                int ch = PMT_CHANNEL_MAP[pmt];
                if (mu1[pmt] <= 0) continue;
                sumPE += area[ch] / mu1[pmt];
            }
            histMuonSpectrum->Fill(sumPE);
            muonCount++;
            
            Long64_t muonTime = nsTime;
            
            // Look for subsequent Michel electrons
            for (Long64_t nextEntry = entry + 1; nextEntry < nEntries; nextEntry++) {
                analysisChain->GetEntry(nextEntry);
                
                // Skip beam events and events too close to beam
                if (triggerBits == 1) {
                    lastBeamTime = nsTime;
                    continue;
                }
                if ((nsTime - lastBeamTime) < 10000) continue;
                
                // Only process specific trigger types
                if (triggerBits != 34 && triggerBits != 2) continue;
                
                // Calculate time difference
                double deltaT = (nsTime - muonTime) * 1e-3;
                
                // Fill accidental background histogram (10-20 μs window)
                if(deltaT >= 10 && deltaT <= 20) {
                    histAccidental->Fill(deltaT);
                    
                    // Reconstruct Michel electron energy for accidental background
                    int michelPMTCount = 0;
                    double michelEnergy = 0.0;
                    for (int pmt = 0; pmt < N_PMTS; pmt++) {
                        double pe = area[PMT_CHANNEL_MAP[pmt]] / mu1[pmt];
                        pe *= (1.0 - CROSS_TALK_PROBABILITY);
                        pe /= (1.0 + AFTERPULSE_PROBABILITY);
                        
                        if (pe >= 2) {
                            michelPMTCount++;
                            michelEnergy += pe;
                        }
                    }
                    if (michelPMTCount >= 6) {
                        histMichelSpectrum_Accidental->Fill(michelEnergy);
                    }
                }
                
                // Stop searching if beyond 20 μs
                if (deltaT > 20) break;
                if (deltaT < 1) continue;
                
                // Apply main event selection to Michel candidate
                if (!passesMainEventSelection(pulseH, baselineRMS, area, peakPosition, mu1)) continue;
                
                // Reconstruct Michel electron energy
                int michelPMTCount = 0;
                double michelEnergy = 0.0;
                for (int pmt = 0; pmt < N_PMTS; pmt++) {
                    double pe = area[PMT_CHANNEL_MAP[pmt]] / mu1[pmt];
                    pe *= (1.0 - CROSS_TALK_PROBABILITY);
                    pe /= (1.0 + AFTERPULSE_PROBABILITY);
                    
                    if (pe >= 2) {
                        michelPMTCount++;
                        michelEnergy += pe;
                    }
                }
                
                // Require at least 6 PMTs hit for Michel electron
                if (michelPMTCount >= 6) {
                    if (deltaT <= 10) {
                        michelCount++;
                        histDeltaT->Fill(deltaT);
                        histMichelSpectrum->Fill(michelEnergy);
                    }
                    break;
                }
            }
        }
    }

    // ==================================================================
    // Print analysis summary
    // ==================================================================
    cout << "\nANALYSIS SUMMARY:" << endl;
    cout << "Total events processed: " << nEntries << endl;
    cout << "TriggerBits==34 events: " << totalTrigger34Events << endl;
    cout << "Muon candidates (>10 PMTs): " << muonCount << endl;
    cout << "Michel electrons: " << michelCount << endl;
    cout << "Michel fraction: " << (muonCount > 0 ? 100.0 * michelCount / muonCount : 0) << "%" << endl;
    
    // ==================================================================
    // Create overlay plot of energy spectra
    // ==================================================================
    TCanvas *cEnergyOverlay = new TCanvas("cEnergyOverlay", "Michel Energy Spectrum Comparison", 1200, 800);
    cEnergyOverlay->SetLeftMargin(0.15);
    cEnergyOverlay->SetRightMargin(0.10);
    cEnergyOverlay->SetBottomMargin(0.15);
    cEnergyOverlay->SetTopMargin(0.10);

    // Remove statistics box for this plot only
    gStyle->SetOptStat(0);

    // Style settings for signal spectrum
    histMichelSpectrum->SetLineColor(kBlue);
    histMichelSpectrum->SetLineWidth(2);
    histMichelSpectrum->SetTitle("Michel Electron Energy Spectrum;Photoelectrons;Counts/10 p.e.");
    histMichelSpectrum->GetYaxis()->SetTitleOffset(1.4);

    // Style settings for accidental spectrum
    histMichelSpectrum_Accidental->SetLineColor(kRed);
    histMichelSpectrum_Accidental->SetLineWidth(2);
    histMichelSpectrum_Accidental->SetLineStyle(2);

    // Draw both histograms
    histMichelSpectrum->Draw("HIST");
    histMichelSpectrum_Accidental->Draw("HIST SAME");

    // Add legend
    TLegend *legEnergy = new TLegend(0.40, 0.75, 0.60, 0.88);
    legEnergy->AddEntry(histMichelSpectrum, "1-10 #mus", "l");
    legEnergy->AddEntry(histMichelSpectrum_Accidental, "10-20 #mus", "l");
    legEnergy->SetBorderSize(0);
    legEnergy->Draw();

    // Save the plot
    cEnergyOverlay->SaveAs((outputDir + "/energy_spectrum_comparison.png").c_str());

    // Restore statistics display for other plots
    gStyle->SetOptStat(1111);

    // ==================================================================
    // Plot accidental background
    // ==================================================================
    TCanvas *cAccidental = new TCanvas("cAccidental", "Accidental Background", 1200, 800);
    histAccidental->SetMarkerStyle(20);
    histAccidental->Draw("PE");
    cAccidental->SaveAs((outputDir + "/accidental_spectrum.png").c_str());
    cAccidental->SetLeftMargin(0.15);
    cAccidental->SetRightMargin(0.10);
    cAccidental->SetBottomMargin(0.15);
    cAccidental->SetTopMargin(0.10);
    
    // ==================================================================
    // Lifetime fitting
    // ==================================================================
    
    // Define fit ranges to test
    vector< pair<double,double> > fitRanges = {
        make_pair(1.0, 10.0), make_pair(1.5, 10.0), make_pair(2.0, 10.0), make_pair(2.5, 10.0)
    };
    
    // Storage for fit results
    vector<double> taus, tau_errors, chi2_ndf;
    vector<double> chi2_values, ndf_values;
    vector<TF1*> fitFunctions;
    
    // Create canvas for time difference fits
    TCanvas *cFits = new TCanvas("cFits", "Time Difference Fits", 1200, 800);
    cFits->SetLeftMargin(0.15);
    cFits->SetRightMargin(0.10);
    cFits->SetBottomMargin(0.15);
    cFits->SetTopMargin(0.10);
    cFits->Divide(2,2);
    
    // Perform fits for each range
    for (size_t i = 0; i < fitRanges.size(); i++) {
        double fitMin = fitRanges[i].first;
        double fitMax = fitRanges[i].second;
        
        // Create exponential fit function
        TF1 *expFit = new TF1(Form("expFit_%zu", i), DecayFit, fitMin, fitMax, 2);
        expFit->SetParameters(histDeltaT->GetBinContent(histDeltaT->GetMaximumBin()), 2.2);
        expFit->SetParLimits(0, 0, histDeltaT->GetBinContent(histDeltaT->GetMaximumBin()) * 2);
        expFit->SetParLimits(1, 0.1, 10);
        
        // Perform fit
        histDeltaT->Fit(expFit, "RQ", "", fitMin, fitMax);
        
        // Store results
        double tau = expFit->GetParameter(1);
        double tauErr = expFit->GetParError(1);
        taus.push_back(tau);
        tau_errors.push_back(tauErr);
        chi2_values.push_back(expFit->GetChisquare());
        ndf_values.push_back(expFit->GetNDF());
        chi2_ndf.push_back(expFit->GetChisquare() / expFit->GetNDF());
        fitFunctions.push_back(expFit);
        
        // Print fit results
        cout << Form("Fit Range: %.1f - %.1f µs", fitMin, fitMax) << endl;
        cout << Form("τ = %.4f ± %.4f µs", tau, tauErr) << endl;
        cout << Form("χ² = %.1f", chi2_values[i]) << endl;
        cout << Form("NDF = %d", (int)ndf_values[i]) << endl;
        cout << Form("χ²/NDF = %.4f", chi2_ndf[i]) << endl;
        cout << "----------------------------------------" << endl;
        
        // Draw fit results
        cFits->cd(i+1);
        histDeltaT->SetMarkerStyle(20);
        histDeltaT->Draw("PE");
        expFit->Draw("same");
        
        // Add fit information box
        TPaveText *pt = new TPaveText(0.60, 0.60, 0.80, 0.75, "NDC");
        pt->SetFillColor(0);
        pt->SetTextSize(0.035);
        pt->SetBorderSize(0);
        pt->AddText(Form("Fit Range: %.1f-%.1f #mus", fitMin, fitMax));
        pt->AddText(Form("#tau = %.4f #pm %.4f #mus", tau, tauErr));
        pt->AddText(Form("#chi^{2}/NDF = %.1f/%d", chi2_values[i], (int)ndf_values[i]));
        pt->Draw("same");
    }
    cFits->SaveAs((outputDir + "/time_difference_fits.png").c_str());
    
    // ==================================================================
    // Create fit summary comparison plot with dual y-axes
    // ==================================================================
    TCanvas *cSummary = new TCanvas("cSummary", "Fit Summary Comparison", 1200, 800);
    cSummary->SetLeftMargin(0.15);
    cSummary->SetRightMargin(0.10);
    cSummary->SetBottomMargin(0.15);
    cSummary->SetTopMargin(0.10);

    // Create and configure tau graph (left y-axis)
    TGraphErrors *gTau = new TGraphErrors(taus.size());
    for (size_t i = 0; i < taus.size(); i++) {
        gTau->SetPoint(i, fitRanges[i].first, taus[i]);
        gTau->SetPointError(i, 0, tau_errors[i]);
    }

    // Clear the default title while keeping axis labels
    gTau->SetTitle(";;");

    gTau->SetMarkerStyle(20);
    gTau->SetMarkerColor(kBlue);
    gTau->SetLineColor(kBlue);
    gTau->GetYaxis()->SetTitle("#tau (#mus)");
    gTau->GetYaxis()->SetTitleOffset(1.2);
    gTau->GetYaxis()->SetTitleColor(kBlue);
    gTau->GetYaxis()->SetLabelColor(kBlue);
    gTau->GetYaxis()->SetAxisColor(kBlue);
    gTau->GetYaxis()->SetRangeUser(*min_element(taus.begin(), taus.end())*0.9, 
                                  *max_element(taus.begin(), taus.end())*1.1);
    gTau->GetXaxis()->SetTitle("Fit Range Lower Bound (#mus)");
    gTau->GetXaxis()->SetTitleOffset(1.2);
    gTau->Draw("APL");

    // Create chi2/NDF graph (right y-axis)
    TGraph *gChi2 = new TGraph(chi2_ndf.size());
    for (size_t i = 0; i < chi2_ndf.size(); i++) {
        gChi2->SetPoint(i, fitRanges[i].first, chi2_ndf[i]);
    }
    gChi2->SetMarkerStyle(21);
    gChi2->SetMarkerColor(kRed);
    gChi2->SetLineColor(kRed);

    // Create transparent pad for second y-axis
    TPad *pad2 = new TPad("pad2", "pad2", 0, 0, 1, 1);
    pad2->SetFillStyle(4000); // Transparent
    pad2->SetFrameFillStyle(0);
    pad2->Range(0,0,1,1);
    pad2->SetLeftMargin(0.15);
    pad2->SetRightMargin(0.10);
    pad2->SetBottomMargin(0.15);
    pad2->SetTopMargin(0.10);
    pad2->Draw();
    pad2->cd();

    // Configure and draw chi2/NDF graph
    double chi2_min = *min_element(chi2_ndf.begin(), chi2_ndf.end());
    double chi2_max = *max_element(chi2_ndf.begin(), chi2_ndf.end());
    gChi2->GetYaxis()->SetRangeUser(chi2_min*0.9, chi2_max*1.1);
    gChi2->GetYaxis()->SetTitle("#chi^{2}/NDF");
    gChi2->GetYaxis()->SetTitleOffset(1.2);
    gChi2->GetYaxis()->SetTitleColor(kRed);
    gChi2->GetYaxis()->SetLabelColor(kRed);
    gChi2->GetYaxis()->SetAxisColor(kRed);
    gChi2->Draw("APL Y+");

    // Add main title
    TLatex *title = new TLatex();
    title->SetTextSize(0.04);
    title->DrawLatexNDC(0.15, 0.92, "Fit Parameter Comparison vs. Fit Range");

    // Add legend
    TLegend *leg = new TLegend(0.65, 0.70, 0.85, 0.85);
    leg->AddEntry(gTau, "#tau (#mus)", "lp");
    leg->AddEntry(gChi2, "#chi^{2}/NDF", "lp");
    leg->SetTextSize(0.035);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->Draw();

    cSummary->SaveAs((outputDir + "/fit_comparison.png").c_str());

    // ==================================================================
    // Utility function to plot histograms with consistent style
    // ==================================================================
    auto plotHistogram = [&outputDir](TH1F* hist, const string& name,
                                      const string& title, bool addCutLine = false, double cutValue = 0) {
        TCanvas *c = new TCanvas(name.c_str(), title.c_str(), 1200, 800);
        c->SetLeftMargin(0.15);
        c->SetRightMargin(0.10);
        c->SetBottomMargin(0.15);
        c->SetTopMargin(0.10);
        
        // Axis title settings
        hist->GetXaxis()->SetTitleOffset(1.2);
        hist->GetYaxis()->SetTitleOffset(1.4);
        
        // Special settings for specific histograms
        if (name == "muon_pmt_hits") {
            hist->GetYaxis()->SetTitleOffset(1.6);
        }
        if (name == "muon_spectrum") {
            hist->GetYaxis()->SetTitleOffset(1.9);
        }
        
        // Label sizes
        hist->GetXaxis()->SetLabelSize(0.045);
        hist->GetYaxis()->SetLabelSize(0.045);
        
        // Special settings for PMT hits histogram
        if (name == "muon_pmt_hits") {
            hist->GetYaxis()->SetNoExponent(false);
            hist->GetYaxis()->SetMaxDigits(3);
            hist->GetYaxis()->SetNoExponent(kTRUE);
            hist->GetYaxis()->SetNoExponent(kFALSE);
        }
        
        // Draw histogram
        hist->Draw();
        
        // Save plot
        c->SaveAs((outputDir + "/" + name + ".png").c_str());
        delete c;
    };
    
    // ==================================================================
    // Save all analysis histograms
    // ==================================================================
    plotHistogram(histMichelSpectrum, "michel_spectrum", "Michel Electron Energy Spectrum");
    plotHistogram(histMuonPMTHits, "muon_pmt_hits", "PMT Multiplicity");
    plotHistogram(histSiPMMultiplicity, "sipm_multiplicity", "SiPM Multiplicity Distribution", true, 2.5);
    plotHistogram(histTriggerBits, "trigger_bits", "Trigger Bits Distribution");
    plotHistogram(histMuonSpectrum, "muon_spectrum", "Muon Spectrum ");
    
    // ==================================================================
    // Clean up memory
    // ==================================================================
    delete legEnergy;
    delete cEnergyOverlay;
    delete leg;
    delete gTau;
    delete gChi2;
    delete pad2;
    delete cSummary;
    delete cFits;
    delete cAccidental;
    delete histDeltaT;
    delete histMichelSpectrum;
    delete histMichelSpectrum_Accidental;
    delete histMuonPMTHits;
    delete histSiPMMultiplicity;
    delete histTriggerBits;
    delete histMuonSpectrum;
    delete histAccidental;
}


// ======================================================================
// Main Program
// ======================================================================

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <calibration_file.root> <analysis_file1.root> [analysis_file2.root ...]" << endl;
        cerr << "Note: First file is used ONLY for calibration (triggerBits==16), others for analysis" << endl;
        return 1;
    }
    
    // Create output directory
    string outputDir = generateOutputDirName();
    cout << "Creating output directory: " << outputDir << endl;
    mkdir(outputDir.c_str(), 0777);
    
    // Arrays for calibration results
    Double_t mu1[N_PMTS], mu1_err[N_PMTS];
    
    // Perform PMT calibration
    cout << "\n=== Performing SPE Calibration ===" << endl;
    cout << "Using calibration file: " << argv[1] << endl;
    performCalibration(argv[1], mu1, mu1_err);
    
    // Set up analysis chain
    TChain *analysisChain = new TChain("tree");
    cout << "\n=== Setting Up Analysis Chain ===" << endl;
    for (int i = 2; i < argc; i++) {
        analysisChain->Add(argv[i]);
        cout << "Added analysis file: " << argv[i] << endl;
    }
    
    // Run main analysis
    cout << "\n=== Starting Analysis ===" << endl;
    analyzeMuonMichel(analysisChain, mu1, outputDir);
    
    // Clean up and exit
    delete analysisChain;
    cout << "\nAnalysis complete. Results in: " << outputDir << endl;
    return 0;
}
