import SwiftUI

struct SeasonSimulationView: View {
    @EnvironmentObject var apiClient: APIClient
    @State private var selectedCompetition = "PL"
    @State private var simulation: SeasonSimulationResponse?
    @State private var isLoading = false
    @State private var error: String?

    let competitions = ["PL", "PD", "SA", "BL1", "FL1"]
    let competitionNames: [String: String] = [
        "PL": "Premier League",
        "PD": "La Liga",
        "SA": "Serie A",
        "BL1": "Bundesliga",
        "FL1": "Ligue 1"
    ]

    var sortedTeams: [SimulatedTeam] {
        simulation?.teams.sorted { $0.currentPos < $1.currentPos } ?? []
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Color.appBackground.ignoresSafeArea()

                VStack(spacing: 0) {
                    // Competition picker
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(competitions, id: \.self) { comp in
                                Button(action: {
                                    let impactFeedback = UIImpactFeedbackGenerator(style: .light)
                                    impactFeedback.impactOccurred()
                                    selectedCompetition = comp
                                    loadData()
                                }) {
                                    Text(competitionNames[comp] ?? comp)
                                        .font(.caption)
                                        .padding(.horizontal, 14)
                                        .padding(.vertical, 8)
                                        .background(selectedCompetition == comp ? Color.accentBlue : Color.cardBackground)
                                        .foregroundColor(selectedCompetition == comp ? .white : .primary)
                                        .cornerRadius(8)
                                }
                            }
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 12)
                    }

                    if isLoading {
                        Spacer()
                        VStack(spacing: 12) {
                            ProgressView()
                                .scaleEffect(1.2)
                            Text("Simulating season...")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                        Spacer()
                    } else if let error = error {
                        Spacer()
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
                        Spacer()
                    } else if sortedTeams.isEmpty {
                        Spacer()
                        VStack(spacing: 12) {
                            Image(systemName: "chart.line.uptrend.xyaxis")
                                .font(.system(size: 40))
                                .foregroundColor(.gray)
                            Text("No simulation data")
                                .font(.headline)
                            Text("Select a competition to view season projections")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                        Spacer()
                    } else {
                        // Header row
                        HStack(spacing: 4) {
                            Text("#")
                                .font(.caption2)
                                .fontWeight(.bold)
                                .frame(width: 24, alignment: .center)
                            Text("Team")
                                .font(.caption2)
                                .fontWeight(.bold)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Text("Pts")
                                .font(.caption2)
                                .fontWeight(.bold)
                                .frame(width: 30, alignment: .center)
                            Text("Title")
                                .font(.caption2)
                                .fontWeight(.bold)
                                .foregroundColor(.successGreen)
                                .frame(width: 55, alignment: .center)
                            Text("Top 4")
                                .font(.caption2)
                                .fontWeight(.bold)
                                .foregroundColor(.accentBlue)
                                .frame(width: 55, alignment: .center)
                            Text("Releg.")
                                .font(.caption2)
                                .fontWeight(.bold)
                                .foregroundColor(.dangerRed)
                                .frame(width: 55, alignment: .center)
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                        .background(Color.cardBackground.opacity(0.5))

                        ScrollView {
                            LazyVStack(spacing: 2) {
                                ForEach(sortedTeams) { team in
                                    NavigationLink(destination: TeamProfileView(teamName: team.team)) {
                                        SimulationRowView(team: team)
                                    }
                                }
                            }
                            .padding(.horizontal)
                            .padding(.top, 4)
                        }
                        .refreshable {
                            await refreshData()
                        }
                    }
                }
            }
            .navigationTitle("Season Simulation")
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
            let response = try await apiClient.fetchSeasonSimulation(comp: selectedCompetition)
            await MainActor.run {
                simulation = response
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

struct SimulationRowView: View {
    let team: SimulatedTeam

    var body: some View {
        HStack(spacing: 4) {
            Text("\(team.currentPos)")
                .font(.caption)
                .fontWeight(.semibold)
                .frame(width: 24, alignment: .center)
                .foregroundColor(.primary)

            Text(team.team)
                .font(.caption)
                .fontWeight(.medium)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .foregroundColor(.primary)

            Text("\(team.currentPts)")
                .font(.caption)
                .fontWeight(.semibold)
                .frame(width: 30, alignment: .center)
                .foregroundColor(.primary)

            MiniProbBar(value: team.pTitle, color: .successGreen)
                .frame(width: 55)

            MiniProbBar(value: team.pTop4, color: .accentBlue)
                .frame(width: 55)

            MiniProbBar(value: team.pRelegation, color: .dangerRed)
                .frame(width: 55)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 8)
        .background(Color.cardBackground)
        .cornerRadius(8)
    }
}

struct MiniProbBar: View {
    let value: Double
    let color: Color

    var body: some View {
        VStack(spacing: 2) {
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(color.opacity(0.15))
                        .frame(height: 6)

                    RoundedRectangle(cornerRadius: 2)
                        .fill(color)
                        .frame(width: max(0, geometry.size.width * CGFloat(min(value, 1.0))), height: 6)
                }
            }
            .frame(height: 6)

            Text(String(format: "%.0f%%", value * 100))
                .font(.system(size: 9, weight: .semibold))
                .foregroundColor(color)
        }
    }
}

#Preview {
    SeasonSimulationView()
        .environmentObject(APIClient())
}
