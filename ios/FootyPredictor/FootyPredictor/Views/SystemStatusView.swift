import SwiftUI

struct SystemStatusView: View {
    @EnvironmentObject var apiClient: APIClient
    @State private var trainingStatus: TrainingStatusResponse?
    @State private var modelLab: ModelLabResponse?
    @State private var selfLearning: SelfLearningStatusResponse?
    @State private var health: HealthResponse?
    @State private var isLoading = false
    @State private var error: String?
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.appBackground.ignoresSafeArea()
                
                if isLoading {
                    VStack {
                        ProgressView()
                            .scaleEffect(1.2)
                        Text("Loading system status...")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                } else if let error = error {
                    VStack(spacing: 12) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.largeTitle)
                            .foregroundColor(.warningYellow)
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.gray)
                            .multilineTextAlignment(.center)
                        
                        Button(action: loadData) {
                            Text("Try Again")
                                .font(.caption)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(Color.accentBlue)
                                .foregroundColor(.white)
                                .cornerRadius(6)
                        }
                    }
                    .padding()
                } else {
                    ScrollView {
                        VStack(spacing: 16) {
                            if let health = health {
                                HealthCardView(health: health)
                            }
                            
                            if let training = trainingStatus {
                                TrainingCardView(status: training)
                            }
                            
                            if let selfLearning = selfLearning {
                                SelfLearningCardView(status: selfLearning)
                            }
                            
                            if let modelLab = modelLab {
                                ModelLabCardView(lab: modelLab)
                            }
                        }
                        .padding()
                    }
                    .refreshable { await loadDataAsync() }
                }
            }
            .navigationTitle("System Status")
            .onAppear { loadData() }
        }
    }
    
    private func loadData() {
        isLoading = true
        error = nil
        
        Task {
            await loadDataAsync()
        }
    }
    
    @MainActor
    private func loadDataAsync() async {
        isLoading = true
        error = nil
        
        // Fetch each independently so one failure doesn't block others
        async let trainingResult = Result { try await apiClient.fetchTrainingStatus() }
        async let modelLabResult = Result { try await apiClient.fetchModelLab() }
        async let selfLearningResult = Result { try await apiClient.fetchSelfLearningStatus() }
        async let healthResult = Result { try await apiClient.fetchHealth() }

        let (tr, ml, sl, he) = await (trainingResult, modelLabResult, selfLearningResult, healthResult)

        if case .success(let v) = tr {
            self.trainingStatus = v
            CacheManager.shared.cache(v, forKey: CacheManager.trainingStatusCacheKey)
        }
        if case .success(let v) = ml {
            self.modelLab = v
            CacheManager.shared.cache(v, forKey: CacheManager.modelLabCacheKey)
        }
        if case .success(let v) = sl {
            self.selfLearning = v
            CacheManager.shared.cache(v, forKey: CacheManager.selfLearningCacheKey)
        }
        if case .success(let v) = he {
            self.health = v
            CacheManager.shared.cache(v, forKey: CacheManager.healthCacheKey)
        }

        // Only show error if ALL calls failed
        let allFailed = [tr, ml, sl, he].allSatisfy { if case .failure = $0 { return true }; return false }
        if allFailed {
            if case .failure(let err) = tr {
                self.error = err.localizedDescription
            }
        }
        self.isLoading = false
    }
}

struct HealthCardView: View {
    let health: HealthResponse
    
    var statusColor: Color {
        switch health.status.lowercased() {
        case "healthy", "ok":
            return .successGreen
        case "degraded":
            return .warningYellow
        case "error":
            return .dangerRed
        default:
            return .gray
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                HStack(spacing: 8) {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 12, height: 12)
                    Text("System Health")
                        .font(.headline)
                }
                Spacer()
                Text(health.status)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(statusColor)
            }
            
            VStack(spacing: 8) {
                if let app = health.app {
                    StatusRowView(label: "App", value: app)
                }
                if let version = health.version {
                    StatusRowView(label: "Version", value: version)
                }
                if let timestamp = health.timestamp {
                    StatusRowView(label: "Timestamp", value: timestamp)
                }
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
    }
}

struct TrainingCardView: View {
    let status: TrainingStatusResponse
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Training Status")
                .font(.headline)
            
            HStack {
                Text("Status")
                    .font(.caption)
                    .foregroundColor(.gray)
                Spacer()
                Text(status.status)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(.accentBlue)
            }

            if let activeVersion = status.activeVersion {
                HStack {
                    Text("Active Version")
                        .font(.caption)
                        .foregroundColor(.gray)
                    Spacer()
                    Text(activeVersion)
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
            }

            if let lastTraining = status.lastTraining {
                HStack {
                    Text("Last Training")
                        .font(.caption)
                        .foregroundColor(.gray)
                    Spacer()
                    Text(lastTraining)
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
            }

            if let nextScheduled = status.nextScheduled {
                HStack {
                    Text("Next Scheduled")
                        .font(.caption)
                        .foregroundColor(.gray)
                    Spacer()
                    Text(nextScheduled)
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
            }

            if let message = status.message {
                HStack {
                    Text("Message")
                        .font(.caption)
                        .foregroundColor(.gray)
                    Spacer()
                    Text(message)
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
    }
}

struct SelfLearningCardView: View {
    let status: SelfLearningStatusResponse

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Self-Learning")
                    .font(.headline)
                Spacer()
                HStack(spacing: 6) {
                    Circle()
                        .fill(status.learning ? Color.successGreen : Color.gray)
                        .frame(width: 8, height: 8)
                    Text(status.learning ? "Active" : "Inactive")
                        .font(.caption)
                        .fontWeight(.semibold)
                }
            }

            HStack {
                Text("Status")
                    .font(.caption)
                    .foregroundColor(.gray)
                Spacer()
                Text(status.status)
                    .font(.caption2)
                    .fontWeight(.semibold)
                    .foregroundColor(.accentBlue)
            }

            if let report = status.report {
                if let overall = report.overall {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Performance")
                            .font(.caption)
                            .foregroundColor(.gray)

                        if let accuracy = overall.accuracy {
                            HStack {
                                Text("Accuracy")
                                    .font(.caption2)
                                Spacer()
                                Text(String(format: "%.2f%%", accuracy * 100))
                                    .font(.caption2)
                                    .fontWeight(.semibold)
                                    .foregroundColor(.successGreen)
                            }
                        }

                        if let logLoss = overall.meanLogLoss {
                            HStack {
                                Text("Log Loss")
                                    .font(.caption2)
                                Spacer()
                                Text(String(format: "%.3f", logLoss))
                                    .font(.caption2)
                                    .fontWeight(.semibold)
                                    .foregroundColor(.accentBlue)
                            }
                        }

                        if let n = overall.nPredictions {
                            HStack {
                                Text("Predictions")
                                    .font(.caption2)
                                Spacer()
                                Text("\(n)")
                                    .font(.caption2)
                                    .fontWeight(.semibold)
                            }
                        }
                    }
                }

                if let driftDetected = report.driftDetected, driftDetected {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.warningYellow)
                        Text("Drift Detected")
                            .font(.caption)
                            .foregroundColor(.warningYellow)
                    }
                }
            }

            if let message = status.message {
                Text(message)
                    .font(.caption2)
                    .foregroundColor(.gray)
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
    }
}

struct ModelLabCardView: View {
    let lab: ModelLabResponse

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Model Lab")
                .font(.headline)

            if let activeVersion = lab.activeVersion {
                HStack {
                    Text("Active Version")
                        .font(.caption)
                        .foregroundColor(.gray)
                    Spacer()
                    Text(activeVersion)
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundColor(.accentBlue)
                }
            }

            if !lab.models.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Available Models")
                        .font(.caption)
                        .foregroundColor(.gray)
                    ForEach(lab.models, id: \.self) { model in
                        HStack {
                            Image(systemName: "circle.fill")
                                .font(.system(size: 4))
                                .foregroundColor(.accentBlue)
                            Text(model)
                                .font(.caption2)
                                .fontWeight(.semibold)
                        }
                    }
                }
            }

            if let weights = lab.ensembleWeights, !weights.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Ensemble Weights")
                        .font(.caption)
                        .foregroundColor(.gray)

                    ForEach(weights.prefix(10), id: \.model) { entry in
                        HStack {
                            Text(entry.model)
                                .font(.caption2)
                                .lineLimit(1)
                            Spacer()
                            Text(String(format: "%.4f", entry.weight))
                                .font(.caption2)
                                .fontWeight(.semibold)
                                .foregroundColor(.accentBlue)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
    }
}

struct StatusRowView: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.gray)
            Spacer()
            Text(value)
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundColor(.accentBlue)
        }
    }
}

#Preview {
    SystemStatusView()
        .environmentObject(APIClient())
}
