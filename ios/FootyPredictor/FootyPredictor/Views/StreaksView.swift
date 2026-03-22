import SwiftUI

struct StreaksView: View {
    @EnvironmentObject var apiClient: APIClient
    @State private var streaks: StreaksResponse?
    @State private var isLoading = false
    @State private var error: String?

    var body: some View {
        NavigationStack {
            ZStack {
                Color.appBackground.ignoresSafeArea()

                if isLoading {
                    VStack(spacing: 12) {
                        ProgressView()
                            .scaleEffect(1.2)
                        Text("Loading streaks...")
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
                } else if let streaks = streaks {
                    ScrollView {
                        VStack(spacing: 20) {
                            if let winning = streaks.winningStreaks, !winning.isEmpty {
                                StreakSectionView(
                                    title: "Winning Streaks",
                                    icon: "flame.fill",
                                    streaks: winning,
                                    color: .successGreen
                                )
                            }

                            if let losing = streaks.losingStreaks, !losing.isEmpty {
                                StreakSectionView(
                                    title: "Losing Streaks",
                                    icon: "arrow.down.circle.fill",
                                    streaks: losing,
                                    color: .dangerRed
                                )
                            }

                            if let unbeaten = streaks.unbeatenRuns, !unbeaten.isEmpty {
                                StreakSectionView(
                                    title: "Unbeaten Runs",
                                    icon: "shield.fill",
                                    streaks: unbeaten,
                                    color: .accentBlue
                                )
                            }

                            if streaks.winningStreaks?.isEmpty ?? true &&
                               streaks.losingStreaks?.isEmpty ?? true &&
                               streaks.unbeatenRuns?.isEmpty ?? true {
                                VStack(spacing: 12) {
                                    Image(systemName: "chart.bar.xaxis")
                                        .font(.system(size: 40))
                                        .foregroundColor(.gray)
                                    Text("No streak data available")
                                        .font(.headline)
                                    Text("Check back later for streak information")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(40)
                            }

                            Spacer(minLength: 40)
                        }
                        .padding()
                    }
                    .refreshable { await refreshData() }
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "chart.bar.xaxis")
                            .font(.system(size: 40))
                            .foregroundColor(.gray)
                        Text("No streak data")
                            .font(.headline)
                    }
                }
            }
            .navigationTitle("Streaks")
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

    private func loadDataAsync() async {
        await MainActor.run {
            isLoading = true
            error = nil
        }

        do {
            let response = try await apiClient.fetchStreaks()
            await MainActor.run {
                streaks = response
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

struct StreakSectionView: View {
    let title: String
    let icon: String
    let streaks: [TeamStreak]
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .foregroundColor(color)
                Text(title)
                    .font(.headline)
            }

            VStack(spacing: 6) {
                ForEach(streaks.sorted { $0.streak > $1.streak }) { streak in
                    HStack(spacing: 12) {
                        Text(streak.team)
                            .font(.caption)
                            .fontWeight(.semibold)
                            .lineLimit(1)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        if let comp = streak.competition {
                            Text(comp)
                                .font(.caption2)
                                .foregroundColor(.gray)
                        }

                        // Streak bar
                        HStack(spacing: 2) {
                            ForEach(0..<min(streak.streak, 15), id: \.self) { _ in
                                RoundedRectangle(cornerRadius: 1)
                                    .fill(color)
                                    .frame(width: 6, height: 16)
                            }
                        }

                        Text("\(streak.streak)")
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundColor(color)
                            .frame(width: 24, alignment: .trailing)
                    }
                    .padding(.vertical, 8)
                    .padding(.horizontal, 12)
                    .background(color.opacity(0.06))
                    .cornerRadius(8)
                }
            }
        }
        .padding()
        .background(Color.cardBackground)
        .cornerRadius(12)
    }
}

#Preview {
    StreaksView()
        .environmentObject(APIClient())
}
