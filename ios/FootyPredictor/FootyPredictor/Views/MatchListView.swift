import SwiftUI

struct MatchListView: View {
    @EnvironmentObject var apiClient: APIClient
    @State private var matches: [Match] = []
    @State private var isLoading = false
    @State private var error: String?
    @State private var selectedDays = 7
    @State private var selectedCompetition: String?
    @State private var searchText = ""
    @State private var currentPage = 1
    @State private var hasMore = false
    
    let competitions = ["PL", "PD", "SA", "BL1", "FL1", "DED"]
    let dayOptions = [3, 7, 14, 30]
    
    var filteredMatches: [Match] {
        matches.filter { match in
            if let competition = selectedCompetition, !competition.isEmpty {
                if match.competition != competition { return false }
            }
            if !searchText.isEmpty {
                return match.homeTeam.localizedCaseInsensitiveContains(searchText) ||
                       match.awayTeam.localizedCaseInsensitiveContains(searchText)
            }
            return true
        }
    }
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(dayOptions, id: \.self) { days in
                            Button(action: { selectedDays = days; loadMatches() }) {
                                Text("\(days)d")
                                    .font(.caption)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(selectedDays == days ? Color.accentBlue : Color.cardBackground)
                                    .foregroundColor(selectedDays == days ? .white : .primary)
                                    .cornerRadius(6)
                            }
                        }
                    }
                    .padding(.horizontal)
                }
                
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        Button(action: { selectedCompetition = nil }) {
                            Text("All")
                                .font(.caption)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(selectedCompetition == nil ? Color.accentBlue : Color.cardBackground)
                                .foregroundColor(selectedCompetition == nil ? .white : .primary)
                                .cornerRadius(6)
                        }
                        
                        ForEach(competitions, id: \.self) { comp in
                            Button(action: { selectedCompetition = comp }) {
                                Text(comp)
                                    .font(.caption)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(selectedCompetition == comp ? Color.accentBlue : Color.cardBackground)
                                    .foregroundColor(selectedCompetition == comp ? .white : .primary)
                                    .cornerRadius(6)
                            }
                        }
                    }
                    .padding(.horizontal)
                }
                
                if let error = error {
                    VStack {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.largeTitle)
                            .foregroundColor(.warningYellow)
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.gray)
                            .multilineTextAlignment(.center)
                        
                        Button(action: { loadMatches() }) {
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
                    .frame(maxWidth: .infinity)
                    .background(Color.cardBackground)
                    .cornerRadius(8)
                    .padding()
                    Spacer()
                } else if isLoading {
                    VStack {
                        ProgressView()
                            .scaleEffect(1.2)
                        Text("Loading matches...")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(40)
                    Spacer()
                } else if filteredMatches.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "rectangle.and.text.magnifyingglass")
                            .font(.system(size: 40))
                            .foregroundColor(.gray)
                        Text("No matches found")
                            .font(.headline)
                        Text("Try adjusting your filters")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(40)
                    Spacer()
                } else {
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(filteredMatches) { match in
                                NavigationLink(destination: MatchDetailView(matchId: match.matchId)) {
                                    MatchCardView(match: match, onTap: {
                                        let impactFeedback = UIImpactFeedbackGenerator(style: .light)
                                        impactFeedback.impactOccurred()
                                    })
                                }
                            }
                            
                            if hasMore {
                                Button(action: { loadMore() }) {
                                    HStack {
                                        Spacer()
                                        Text("Load More")
                                            .font(.caption)
                                            .foregroundColor(.accentBlue)
                                        Spacer()
                                    }
                                    .padding()
                                    .background(Color.cardBackground)
                                    .cornerRadius(8)
                                }
                            }
                        }
                        .padding()
                    }
                }
            }
            .background(Color.appBackground)
            .navigationTitle("Predictions")
            .searchable(text: $searchText, prompt: "Search teams...")
            .refreshable {
                let impactFeedback = UIImpactFeedbackGenerator(style: .light)
                impactFeedback.impactOccurred()
                await refreshData()
            }
            .onAppear { loadMatches() }
        }
    }
    
    private func loadMatches() {
        currentPage = 1
        matches = []
        loadMore()
    }

    private func loadMore() {
        isLoading = true
        error = nil

        Task {
            do {
                let response = try await apiClient.fetchMatches(
                    days: selectedDays,
                    competition: selectedCompetition,
                    page: currentPage,
                    limit: 50
                )

                await MainActor.run {
                    if currentPage == 1 {
                        matches = response.matches
                    } else {
                        matches.append(contentsOf: response.matches)
                    }
                    hasMore = response.hasMore
                    currentPage += 1
                    isLoading = false
                }

                CacheManager.shared.cache(response, forKey: CacheManager.matchListCacheKey)
            } catch {
                await MainActor.run {
                    self.error = error.localizedDescription
                    isLoading = false

                    if matches.isEmpty {
                        if let cached: MatchListResponse = CacheManager.shared.retrieveCache(
                            forKey: CacheManager.matchListCacheKey,
                            type: MatchListResponse.self
                        ) {
                            matches = cached.matches
                            self.error = "Showing cached data. \(error.localizedDescription)"
                        }
                    }
                }
            }
        }
    }

    private func refreshData() async {
        await withCheckedContinuation { continuation in
            currentPage = 1
            matches = []
            isLoading = true
            error = nil

            Task {
                do {
                    let response = try await apiClient.fetchMatches(
                        days: selectedDays,
                        competition: selectedCompetition,
                        page: 1,
                        limit: 50
                    )

                    await MainActor.run {
                        matches = response.matches
                        hasMore = response.hasMore
                        currentPage = 2
                        isLoading = false
                    }

                    CacheManager.shared.cache(response, forKey: CacheManager.matchListCacheKey)
                } catch {
                    await MainActor.run {
                        self.error = error.localizedDescription
                        isLoading = false

                        if matches.isEmpty {
                            if let cached: MatchListResponse = CacheManager.shared.retrieveCache(
                                forKey: CacheManager.matchListCacheKey,
                                type: MatchListResponse.self
                            ) {
                                matches = cached.matches
                                self.error = "Showing cached data. \(error.localizedDescription)"
                            }
                        }
                    }
                }
                continuation.resume()
            }
        }
    }
}

struct SearchBar: View {
    @Binding var text: String
    let placeholder: String
    
    var body: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.gray)
            
            TextField(placeholder, text: $text)
                .textFieldStyle(.roundedBorder)
            
            if !text.isEmpty {
                Button(action: { text = "" }) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.gray)
                }
            }
        }
        .padding(.horizontal)
    }
}

#Preview {
    MatchListView()
        .environmentObject(APIClient())
}
