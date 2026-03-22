import SwiftUI

struct InsightsView: View {
    @EnvironmentObject var apiClient: APIClient
    @State private var selectedTab = 0

    var body: some View {
        NavigationStack {
            ZStack {
                Color.appBackground.ignoresSafeArea()

                VStack(spacing: 0) {
                    Picker("Insights", selection: $selectedTab) {
                        Text("Value Bets").tag(0)
                        Text("BTTS/O2.5").tag(1)
                        Text("Accumulators").tag(2)
                    }
                    .pickerStyle(.segmented)
                    .padding()

                    Group {
                        if selectedTab == 0 {
                            ValueBetsTabView(apiClient: apiClient)
                        } else if selectedTab == 1 {
                            BTTSOUTabView(apiClient: apiClient)
                        } else {
                            AccumulatorsTabView(apiClient: apiClient)
                        }
                    }

                    Spacer()
                }
            }
            .navigationTitle("Insights")
        }
    }
}

struct ValueBetsTabView: View {
    @ObservedObject var apiClient: APIClient
    @State private var bets: [ValueBet] = []
    @State private var isLoading = false
    @State private var error: String?

    var body: some View {
        Group {
            if isLoading {
                VStack {
                    ProgressView()
                        .scaleEffect(1.2)
                    Text("Loading value bets...")
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
            } else if bets.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "rectangle.and.text.magnifyingglass")
                        .font(.title2)
                        .foregroundColor(.gray)
                    Text("No value bets found")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                .frame(maxWidth: .infinity)
                .padding(20)
            } else {
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(bets) { bet in
                            ValueBetCardView(bet: bet)
                        }
                    }
                    .padding()
                }
            }
        }
        .onAppear { onAppear() }
    }

    private func loadData() {
        isLoading = true
        error = nil

        Task {
            do {
                let response = try await apiClient.fetchValueBets()
                await MainActor.run {
                    bets = response.bets
                    isLoading = false
                    CacheManager.shared.cache(response, forKey: CacheManager.valueBetsCacheKey)
                }
            } catch {
                await MainActor.run {
                    self.error = error.localizedDescription
                    isLoading = false

                    if bets.isEmpty {
                        if let cached: ValueBetsResponse = CacheManager.shared.retrieveCache(
                            forKey: CacheManager.valueBetsCacheKey,
                            type: ValueBetsResponse.self
                        ) {
                            bets = cached.bets
                            self.error = "Showing cached data"
                        }
                    }
                }
            }
        }
    }

    private func onAppear() {
        if bets.isEmpty {
            loadData()
        }
    }
}

struct ValueBetCardView: View {
    let bet: ValueBet

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(bet.homeTeam) vs \(bet.awayTeam)")
                        .font(.headline)
                    Text(bet.competition)
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                Spacer()
                Text(bet.date)
                    .font(.caption2)
                    .foregroundColor(.gray)
            }

            HStack(spacing: 12) {
                VStack(alignment: .center, spacing: 4) {
                    Text("Bet")
                        .font(.caption2)
                        .foregroundColor(.gray)
                    Text(bet.displayBet)
                        .font(.caption)
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.accentBlue.opacity(0.1))
                .cornerRadius(6)

                VStack(alignment: .center, spacing: 4) {
                    Text("Model Prob")
                        .font(.caption2)
                        .foregroundColor(.gray)
                    Text(String(format: "%.0f%%", bet.modelProb * 100))
                        .font(.caption)
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.successGreen.opacity(0.1))
                .cornerRadius(6)

                VStack(alignment: .center, spacing: 4) {
                    Text("Edge")
                        .font(.caption2)
                        .foregroundColor(.gray)
                    Text(bet.edgePercentage)
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.successGreen)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.successGreen.opacity(0.1))
                .cornerRadius(6)
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
    }
}

struct BTTSOUTabView: View {
    @ObservedObject var apiClient: APIClient
    @State private var data: BTTSOUResponse?
    @State private var isLoading = false
    @State private var error: String?

    var body: some View {
        Group {
            if isLoading {
                VStack {
                    ProgressView()
                        .scaleEffect(1.2)
                    Text("Loading insights...")
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
            } else if let data = data {
                ScrollView {
                    VStack(spacing: 20) {
                        if !data.bttsLikely.isEmpty {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("BTTS Likely")
                                    .font(.headline)
                                    .padding(.horizontal)

                                ForEach(data.bttsLikely) { match in
                                    InsightMatchCard(title: "BTTS", prob: match.bttsProb, homeTeam: match.homeTeam, awayTeam: match.awayTeam, competition: match.competition)
                                }
                            }
                        }

                        if !data.bttsUnlikely.isEmpty {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("BTTS Unlikely")
                                    .font(.headline)
                                    .padding(.horizontal)

                                ForEach(data.bttsUnlikely) { match in
                                    InsightMatchCard(title: "No BTTS", prob: match.bttsProb, homeTeam: match.homeTeam, awayTeam: match.awayTeam, competition: match.competition)
                                }
                            }
                        }

                        if !data.over25.isEmpty {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("Over 2.5 Goals")
                                    .font(.headline)
                                    .padding(.horizontal)

                                ForEach(data.over25) { match in
                                    InsightMatchCard(title: "O2.5", prob: match.o25Prob, homeTeam: match.homeTeam, awayTeam: match.awayTeam, competition: match.competition)
                                }
                            }
                        }

                        if !data.under25.isEmpty {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("Under 2.5 Goals")
                                    .font(.headline)
                                    .padding(.horizontal)

                                ForEach(data.under25) { match in
                                    InsightMatchCard(title: "U2.5", prob: match.under25Prob, homeTeam: match.homeTeam, awayTeam: match.awayTeam, competition: match.competition)
                                }
                            }
                        }
                    }
                    .padding()
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "rectangle.and.text.magnifyingglass")
                        .font(.title2)
                        .foregroundColor(.gray)
                    Text("No insights available")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                .frame(maxWidth: .infinity)
                .padding(20)
            }
        }
        .onAppear { onAppear() }
    }

    private func loadData() {
        isLoading = true
        error = nil

        Task {
            do {
                let response = try await apiClient.fetchBTTSOU()
                await MainActor.run {
                    data = response
                    isLoading = false
                    CacheManager.shared.cache(response, forKey: CacheManager.bttsouCacheKey)
                }
            } catch {
                await MainActor.run {
                    self.error = error.localizedDescription
                    isLoading = false
                }
            }
        }
    }

    private func onAppear() {
        if data == nil {
            loadData()
        }
    }
}

struct AccumulatorsTabView: View {
    @ObservedObject var apiClient: APIClient
    @State private var accumulators: [Accumulator] = []
    @State private var isLoading = false
    @State private var error: String?

    var body: some View {
        Group {
            if isLoading {
                VStack {
                    ProgressView()
                        .scaleEffect(1.2)
                    Text("Loading accumulators...")
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
            } else if accumulators.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "rectangle.and.text.magnifyingglass")
                        .font(.title2)
                        .foregroundColor(.gray)
                    Text("No accumulators available")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                .frame(maxWidth: .infinity)
                .padding(20)
            } else {
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(accumulators) { acc in
                            AccumulatorCardView(accumulator: acc)
                        }
                    }
                    .padding()
                }
            }
        }
        .onAppear { onAppear() }
    }

    private func loadData() {
        isLoading = true
        error = nil

        Task {
            do {
                let response = try await apiClient.fetchAccumulators()
                await MainActor.run {
                    accumulators = response.accumulators
                    isLoading = false
                    CacheManager.shared.cache(response, forKey: CacheManager.accumulatorsCacheKey)
                }
            } catch {
                await MainActor.run {
                    self.error = error.localizedDescription
                    isLoading = false
                }
            }
        }
    }

    private func onAppear() {
        if accumulators.isEmpty {
            loadData()
        }
    }
}

struct InsightMatchCard: View {
    let title: String
    let prob: Double?
    let homeTeam: String
    let awayTeam: String
    let competition: String

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("\(homeTeam) vs \(awayTeam)")
                    .font(.caption)
                    .fontWeight(.semibold)
                Text(competition)
                    .font(.caption2)
                    .foregroundColor(.gray)
            }
            Spacer()
            if let prob = prob {
                VStack(alignment: .center, spacing: 2) {
                    Text(title)
                        .font(.caption2)
                        .foregroundColor(.gray)
                    Text(String(format: "%.0f%%", prob * 100))
                        .font(.headline)
                        .foregroundColor(.accentBlue)
                }
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(8)
        .padding(.horizontal)
    }
}

struct AccumulatorCardView: View {
    let accumulator: Accumulator

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Accumulator")
                    .font(.headline)
                Spacer()
                Text(String(format: "%.2f", accumulator.odds))
                    .font(.headline)
                    .foregroundColor(.successGreen)
            }

            VStack(spacing: 8) {
                ForEach(accumulator.legs) { leg in
                    HStack(spacing: 8) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(leg.homeTeam) vs \(leg.awayTeam)")
                                .font(.caption)
                                .fontWeight(.semibold)
                            Text(leg.outcomeDisplay)
                                .font(.caption2)
                                .foregroundColor(.gray)
                        }
                        Spacer()
                        Text(String(format: "%.0f%%", leg.probability * 100))
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.accentBlue)
                    }
                    .padding(.vertical, 8)
                    .padding(.horizontal, 12)
                    .background(Color.cardBackground)
                    .cornerRadius(6)
                }
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
    }
}

#Preview {
    InsightsView()
        .environmentObject(APIClient())
}
