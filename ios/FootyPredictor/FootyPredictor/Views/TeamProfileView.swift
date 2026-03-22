import SwiftUI

struct TeamProfileView: View {
    let teamName: String
    @EnvironmentObject var apiClient: APIClient
    @State private var profile: TeamProfileResponse?
    @State private var isLoading = true
    @State private var error: String?

    var body: some View {
        ZStack {
            Color.appBackground.ignoresSafeArea()

            if isLoading {
                VStack(spacing: 12) {
                    ProgressView()
                        .scaleEffect(1.2)
                    Text("Loading profile...")
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
            } else if let profile = profile {
                ScrollView {
                    VStack(spacing: 16) {
                        // Team name header
                        Text(profile.team)
                            .font(.title2)
                            .fontWeight(.bold)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal)

                        // Elo Rating gauge
                        if let elo = profile.elo {
                            EloGaugeView(elo: elo)
                        }

                        // Form circles
                        if let form = profile.form, !form.isEmpty {
                            FormCirclesView(form: form)
                        }

                        // Goals stats
                        if let scored = profile.goalsScored, let conceded = profile.goalsConceded {
                            GoalsStatsView(scored: scored, conceded: conceded)
                        }

                        // Recent Matches
                        if let recentMatches = profile.recentMatches, !recentMatches.isEmpty {
                            RecentMatchesSection(matches: recentMatches)
                        }

                        // Upcoming Fixtures
                        if let fixtures = profile.upcomingFixtures, !fixtures.isEmpty {
                            UpcomingFixturesSection(fixtures: fixtures)
                        }

                        Spacer(minLength: 40)
                    }
                    .padding(.top)
                }
                .refreshable { await refreshData() }
            }
        }
        .navigationTitle(teamName)
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { loadData() }
    }

    private func loadData() {
        isLoading = true
        error = nil

        Task {
            await loadDataAsync()
        }
    }

    private func loadDataAsync() async {
        await MainActor.run {
            isLoading = true
            error = nil
        }

        do {
            let response = try await apiClient.fetchTeamProfile(teamName: teamName)
            await MainActor.run {
                profile = response
                isLoading = false
            }
        } catch {
            await MainActor.run {
                self.error = error.localizedDescription
                isLoading = false
            }
        }
    }

    private func refreshData() async {
        await loadDataAsync()
    }
}

// MARK: - Elo Gauge

struct EloGaugeView: View {
    let elo: Double

    // Elo typically ranges from ~1000 to ~2100 for club football
    private var normalizedElo: Double {
        let minElo = 1000.0
        let maxElo = 2100.0
        return min(max((elo - minElo) / (maxElo - minElo), 0), 1)
    }

    private var eloColor: Color {
        if normalizedElo > 0.75 { return .successGreen }
        if normalizedElo > 0.5 { return .accentBlue }
        if normalizedElo > 0.25 { return .warningYellow }
        return .dangerRed
    }

    var body: some View {
        VStack(spacing: 12) {
            Text("Elo Rating")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            ZStack {
                // Background arc
                Circle()
                    .trim(from: 0.0, to: 0.75)
                    .stroke(Color.gray.opacity(0.2), style: StrokeStyle(lineWidth: 12, lineCap: .round))
                    .rotationEffect(.degrees(135))
                    .frame(width: 140, height: 140)

                // Filled arc
                Circle()
                    .trim(from: 0.0, to: CGFloat(normalizedElo * 0.75))
                    .stroke(eloColor, style: StrokeStyle(lineWidth: 12, lineCap: .round))
                    .rotationEffect(.degrees(135))
                    .frame(width: 140, height: 140)

                // Elo value
                VStack(spacing: 2) {
                    Text(String(format: "%.0f", elo))
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(eloColor)
                    Text("ELO")
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
            }
            .frame(height: 160)

            // Scale labels
            HStack {
                Text("1000")
                    .font(.caption2)
                    .foregroundColor(.gray)
                Spacer()
                Text("1550")
                    .font(.caption2)
                    .foregroundColor(.gray)
                Spacer()
                Text("2100")
                    .font(.caption2)
                    .foregroundColor(.gray)
            }
            .padding(.horizontal, 30)
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
        .padding(.horizontal)
    }
}

// MARK: - Form Circles

struct FormCirclesView: View {
    let form: [String]

    private func colorForResult(_ result: String) -> Color {
        switch result.uppercased() {
        case "W": return .successGreen
        case "D": return .warningYellow
        case "L": return .dangerRed
        default: return .gray
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent Form")
                .font(.headline)

            HStack(spacing: 8) {
                ForEach(Array(form.prefix(10).enumerated()), id: \.offset) { index, result in
                    VStack(spacing: 4) {
                        Circle()
                            .fill(colorForResult(result))
                            .frame(width: 28, height: 28)
                            .overlay(
                                Text(result.uppercased())
                                    .font(.system(size: 11, weight: .bold))
                                    .foregroundColor(.white)
                            )
                    }
                }
                Spacer()
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
        .padding(.horizontal)
    }
}

// MARK: - Goals Stats

struct GoalsStatsView: View {
    let scored: Int
    let conceded: Int

    var goalDifference: Int { scored - conceded }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Goals")
                .font(.headline)

            HStack(spacing: 16) {
                VStack(spacing: 4) {
                    Text("\(scored)")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(.successGreen)
                    Text("Scored")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.successGreen.opacity(0.1))
                .cornerRadius(8)

                VStack(spacing: 4) {
                    Text("\(conceded)")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(.dangerRed)
                    Text("Conceded")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.dangerRed.opacity(0.1))
                .cornerRadius(8)

                VStack(spacing: 4) {
                    Text(goalDifference >= 0 ? "+\(goalDifference)" : "\(goalDifference)")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(goalDifference >= 0 ? .accentBlue : .warningYellow)
                    Text("GD")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.accentBlue.opacity(0.1))
                .cornerRadius(8)
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
        .padding(.horizontal)
    }
}

// MARK: - Recent Matches

struct RecentMatchesSection: View {
    let matches: [TeamMatch]

    private func resultColor(_ result: String) -> Color {
        switch result.uppercased() {
        case "W": return .successGreen
        case "D": return .warningYellow
        case "L": return .dangerRed
        default: return .gray
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent Matches")
                .font(.headline)

            VStack(spacing: 8) {
                ForEach(matches.prefix(8)) { match in
                    HStack(spacing: 12) {
                        Circle()
                            .fill(resultColor(match.result))
                            .frame(width: 8, height: 8)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(match.opponent)
                                .font(.caption)
                                .fontWeight(.semibold)
                                .lineLimit(1)
                            if let comp = match.competition {
                                Text(comp)
                                    .font(.caption2)
                                    .foregroundColor(.gray)
                            }
                        }

                        Spacer()

                        if let gf = match.goalsFor, let ga = match.goalsAgainst {
                            Text("\(gf) - \(ga)")
                                .font(.caption)
                                .fontWeight(.bold)
                                .foregroundColor(resultColor(match.result))
                        }

                        Text(String(match.date.prefix(10)))
                            .font(.caption2)
                            .foregroundColor(.gray)
                    }
                    .padding(.vertical, 8)
                    .padding(.horizontal, 12)
                    .background(Color.cardBackground.opacity(0.5))
                    .cornerRadius(6)
                }
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
        .padding(.horizontal)
    }
}

// MARK: - Upcoming Fixtures

struct UpcomingFixturesSection: View {
    let fixtures: [TeamFixture]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Upcoming Fixtures")
                .font(.headline)

            VStack(spacing: 8) {
                ForEach(fixtures.prefix(5)) { fixture in
                    HStack(spacing: 12) {
                        Image(systemName: "calendar")
                            .font(.caption)
                            .foregroundColor(.accentBlue)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(fixture.opponent)
                                .font(.caption)
                                .fontWeight(.semibold)
                                .lineLimit(1)
                            HStack(spacing: 8) {
                                if let venue = fixture.venue {
                                    Text(venue)
                                        .font(.caption2)
                                        .foregroundColor(.gray)
                                }
                                if let comp = fixture.competition {
                                    Text(comp)
                                        .font(.caption2)
                                        .foregroundColor(.neutralGray)
                                }
                            }
                        }

                        Spacer()

                        Text(String(fixture.date.prefix(10)))
                            .font(.caption2)
                            .foregroundColor(.gray)
                    }
                    .padding(.vertical, 8)
                    .padding(.horizontal, 12)
                    .background(Color.cardBackground.opacity(0.5))
                    .cornerRadius(6)
                }
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
        .padding(.horizontal)
    }
}

#Preview {
    NavigationStack {
        TeamProfileView(teamName: "Manchester City")
    }
    .environmentObject(APIClient())
}
