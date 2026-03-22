import SwiftUI

struct MatchDetailView: View {
    let matchId: Int
    @EnvironmentObject var apiClient: APIClient
    @State private var matchDetail: MatchDetail?
    @State private var experts: [String: ExpertPrediction] = [:]
    @State private var h2h: H2HResponse?
    @State private var form: FormResponse?
    @State private var isLoading = true
    @State private var error: String?
    @State private var selectedTab = 0
    
    var body: some View {
        ZStack {
            Color.appBackground.ignoresSafeArea()
            
            if isLoading {
                VStack {
                    ProgressView()
                        .scaleEffect(1.2)
                    Text("Loading match details...")
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
                }
                .padding()
            } else if let match = matchDetail {
                ScrollView {
                    VStack(spacing: 16) {
                        HeaderView(match: match)
                        
                        if let prediction = match.prediction {
                            PredictionView(prediction: prediction, homeTeam: match.homeTeam, awayTeam: match.awayTeam)
                        }
                        
                        Picker("Tabs", selection: $selectedTab) {
                            Text("Experts").tag(0)
                            Text("H2H").tag(1)
                            Text("Form").tag(2)
                        }
                        .pickerStyle(.segmented)
                        .padding()
                        
                        Group {
                            if selectedTab == 0 {
                                ExpertListView(experts: experts)
                            } else if selectedTab == 1 {
                                H2HView(h2h: h2h)
                            } else if selectedTab == 2 {
                                FormView(form: form)
                            }
                        }
                        
                        Spacer(minLength: 40)
                    }
                    .padding(.horizontal)
                }
                .refreshable { await loadData() }
            }
        }
        .navigationTitle("Match Details")
        .navigationBarTitleDisplayMode(.inline)
        .task { await loadData() }
    }

    private func loadData() async {
        await MainActor.run {
            isLoading = true
            error = nil
        }

        // Fetch each independently so partial failures don't block other data
        async let detailTask = fetchDetail()
        async let expertsTask = fetchExperts()
        async let h2hTask = fetchH2H()
        async let formTask = fetchForm()

        let (detailResult, expertsResult, h2hResult, formResult) = await (detailTask, expertsTask, h2hTask, formTask)

        var errors: [String] = []

        await MainActor.run {
            switch detailResult {
            case .success(let detail):
                matchDetail = detail
                CacheManager.shared.cache(detail, forKey: CacheManager.matchDetailCacheKeyPrefix + String(matchId))
            case .failure(let err):
                errors.append("Details: \(err.localizedDescription)")
                // Try cache fallback
                if let cached: MatchDetail = CacheManager.shared.retrieveCache(
                    forKey: CacheManager.matchDetailCacheKeyPrefix + String(matchId),
                    type: MatchDetail.self
                ) {
                    matchDetail = cached
                }
            }

            switch expertsResult {
            case .success(let expResponse):
                experts = expResponse.experts
            case .failure(let err):
                errors.append("Experts: \(err.localizedDescription)")
            }

            switch h2hResult {
            case .success(let h2hResp):
                h2h = h2hResp
            case .failure(let err):
                errors.append("H2H: \(err.localizedDescription)")
            }

            switch formResult {
            case .success(let formResp):
                form = formResp
            case .failure(let err):
                errors.append("Form: \(err.localizedDescription)")
            }

            // Only show error if the critical detail fetch failed and we have no data
            if matchDetail == nil && !errors.isEmpty {
                self.error = errors.joined(separator: "\n")
            }

            isLoading = false
        }
    }

    private func fetchDetail() async -> Result<MatchDetail, Error> {
        do {
            let detail = try await apiClient.fetchMatchDetail(matchId: matchId)
            return .success(detail)
        } catch {
            return .failure(error)
        }
    }

    private func fetchExperts() async -> Result<ExpertPredictionsResponse, Error> {
        do {
            let response = try await apiClient.fetchMatchExperts(matchId: matchId)
            return .success(response)
        } catch {
            return .failure(error)
        }
    }

    private func fetchH2H() async -> Result<H2HResponse, Error> {
        do {
            let response = try await apiClient.fetchH2H(matchId: matchId)
            return .success(response)
        } catch {
            return .failure(error)
        }
    }

    private func fetchForm() async -> Result<FormResponse, Error> {
        do {
            let response = try await apiClient.fetchForm(matchId: matchId)
            return .success(response)
        } catch {
            return .failure(error)
        }
    }
}

struct HeaderView: View {
    let match: MatchDetail
    
    var body: some View {
        VStack(spacing: 16) {
            HStack {
                VStack(alignment: .leading) {
                    Text(match.competition)
                        .font(.caption)
                        .foregroundColor(.neutralGray)
                    Text(match.status)
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
                Spacer()
                VStack(alignment: .trailing) {
                    Text(match.utcDate.prefix(10))
                        .font(.caption2)
                        .foregroundColor(.gray)
                    Text(match.utcDate.dropFirst(11).dropLast(3))
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
            }
            
            HStack(spacing: 16) {
                VStack(alignment: .center, spacing: 8) {
                    Text(match.homeTeam)
                        .font(.headline)
                        .lineLimit(1)
                    if let score = match.score {
                        Text("\(score.home)")
                            .font(.system(size: 32, weight: .bold))
                            .foregroundColor(.accentBlue)
                    }
                }
                .frame(maxWidth: .infinity)
                
                VStack(spacing: 4) {
                    if let score = match.score {
                        Text("FT")
                            .font(.caption)
                            .foregroundColor(.gray)
                        Text("-")
                            .font(.headline)
                    } else {
                        Text("vs")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                }
                
                VStack(alignment: .center, spacing: 8) {
                    Text(match.awayTeam)
                        .font(.headline)
                        .lineLimit(1)
                    if let score = match.score {
                        Text("\(score.away)")
                            .font(.system(size: 32, weight: .bold))
                            .foregroundColor(.dangerRed)
                    }
                }
                .frame(maxWidth: .infinity)
            }
            .padding()
            .background(Color.cardBackground)
            .cornerRadius(12)
        }
    }
}

struct PredictionView: View {
    let prediction: Prediction
    let homeTeam: String
    let awayTeam: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Prediction")
                .font(.headline)
            
            ProbabilityBar(
                homeProb: prediction.pHome,
                drawProb: prediction.pDraw,
                awayProb: prediction.pAway,
                homeTeam: homeTeam,
                awayTeam: awayTeam
            )
            
            HStack(spacing: 12) {
                if let btts = prediction.btts {
                    VStack(alignment: .center, spacing: 4) {
                        Text("BTTS")
                            .font(.caption2)
                            .foregroundColor(.gray)
                        Text(String(format: "%.0f%%", btts * 100))
                            .font(.headline)
                            .foregroundColor(.successGreen)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.cardBackground)
                    .cornerRadius(8)
                }
                
                if let o25 = prediction.o25 {
                    VStack(alignment: .center, spacing: 4) {
                        Text("Over 2.5")
                            .font(.caption2)
                            .foregroundColor(.gray)
                        Text(String(format: "%.0f%%", o25 * 100))
                            .font(.headline)
                            .foregroundColor(.accentBlue)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.cardBackground)
                    .cornerRadius(8)
                }
            }
            
            HStack(spacing: 12) {
                if let egHome = prediction.egHome {
                    VStack(alignment: .center, spacing: 4) {
                        Text(homeTeam)
                            .font(.caption2)
                            .foregroundColor(.gray)
                            .lineLimit(1)
                        Text(String(format: "xG %.2f", egHome))
                            .font(.caption)
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.cardBackground)
                    .cornerRadius(8)
                }
                
                if let egAway = prediction.egAway {
                    VStack(alignment: .center, spacing: 4) {
                        Text(awayTeam)
                            .font(.caption2)
                            .foregroundColor(.gray)
                            .lineLimit(1)
                        Text(String(format: "xG %.2f", egAway))
                            .font(.caption)
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.cardBackground)
                    .cornerRadius(8)
                }
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
    }
}

struct ExpertListView: View {
    let experts: [String: ExpertPrediction]
    
    var sortedExperts: [(String, ExpertPrediction)] {
        experts.sorted { $0.value.confidence > $1.value.confidence }
    }
    
    var body: some View {
        if experts.isEmpty {
            VStack(spacing: 8) {
                Image(systemName: "person.2")
                    .font(.title2)
                    .foregroundColor(.gray)
                Text("No expert data available")
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            .frame(maxWidth: .infinity)
            .padding(20)
        } else {
            VStack(spacing: 12) {
                ForEach(sortedExperts, id: \.0) { name, expert in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(name)
                                .font(.subheadline)
                                .fontWeight(.semibold)
                            Spacer()
                            Text(String(format: "%.0f%%", expert.confidence * 100))
                                .font(.caption)
                                .foregroundColor(.accentBlue)
                        }
                        
                        HStack(spacing: 12) {
                            ProbBox(label: "H", value: String(format: "%.0f%%", expert.probs.home * 100), color: .accentBlue)
                            ProbBox(label: "D", value: String(format: "%.0f%%", expert.probs.draw * 100), color: .warningYellow)
                            ProbBox(label: "A", value: String(format: "%.0f%%", expert.probs.away * 100), color: .dangerRed)
                        }
                    }
                    .padding()
                    .background(Color.cardBackground)
                    .cornerRadius(8)
                }
            }
        }
    }
}

struct ProbBox: View {
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.gray)
            Text(value)
                .font(.caption)
                .fontWeight(.semibold)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(color.opacity(0.1))
        .foregroundColor(color)
        .cornerRadius(6)
    }
}

struct H2HView: View {
    let h2h: H2HResponse?
    
    var body: some View {
        if let h2h = h2h, let matches = h2h.h2h, !matches.isEmpty {
            VStack(spacing: 12) {
                ForEach(Array(matches.prefix(5).enumerated()), id: \.offset) { _, match in
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(match.homeTeam)
                                .font(.caption)
                                .fontWeight(.semibold)
                            Text(match.awayTeam)
                                .font(.caption)
                                .fontWeight(.semibold)
                        }
                        Spacer()
                        Text(match.scoreFormatted)
                            .font(.headline)
                        Spacer()
                        Text((match.date ?? "").prefix(10))
                            .font(.caption2)
                            .foregroundColor(.gray)
                    }
                    .padding()
                    .background(Color.cardBackground)
                    .cornerRadius(8)
                }
            }
        } else {
            VStack(spacing: 8) {
                Image(systemName: "clock.arrow.circlepath")
                    .font(.title2)
                    .foregroundColor(.gray)
                Text("No H2H history available")
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            .frame(maxWidth: .infinity)
            .padding(20)
        }
    }
}

struct FormView: View {
    let form: FormResponse?
    
    var body: some View {
        if let form = form {
            VStack(spacing: 16) {
                FormTableView(title: "Home Form", matches: form.homeForm)
                FormTableView(title: "Away Form", matches: form.awayForm)
            }
        } else {
            VStack(spacing: 8) {
                Image(systemName: "chart.bar")
                    .font(.title2)
                    .foregroundColor(.gray)
                Text("No form data available")
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            .frame(maxWidth: .infinity)
            .padding(20)
        }
    }
}

struct FormTableView: View {
    let title: String
    let matches: [FormMatch]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
            
            VStack(spacing: 8) {
                ForEach(Array(matches.prefix(5).enumerated()), id: \.offset) { _, match in
                    HStack(spacing: 12) {
                        Image(systemName: match.resultIcon)
                            .font(.caption)
                            .foregroundColor(.accentBlue)

                        Text(match.opponent)
                            .font(.caption)
                            .lineLimit(1)

                        Spacer()

                        if let homeGoals = match.homeGoals, let awayGoals = match.awayGoals {
                            Text("\(homeGoals)-\(awayGoals)")
                                .font(.caption2)
                                .fontWeight(.semibold)
                        }

                        Text((match.date ?? "").prefix(10))
                            .font(.caption2)
                            .foregroundColor(.gray)
                    }
                    .padding(.vertical, 8)
                    .padding(.horizontal, 12)
                    .background(Color.cardBackground)
                    .cornerRadius(6)
                }
            }
        }
    }
}

#Preview {
    NavigationStack {
        MatchDetailView(matchId: 1)
    }
    .environmentObject(APIClient())
}
