import SwiftUI

struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        ZStack {
            Color.appBackground.ignoresSafeArea()

            TabView(selection: $selectedTab) {
                MatchListView()
                    .tabItem {
                        Label("Predictions", systemImage: "chart.bar.fill")
                    }
                    .tag(0)

                InsightsView()
                    .tabItem {
                        Label("Insights", systemImage: "lightbulb.fill")
                    }
                    .tag(1)

                SeasonSimulationView()
                    .tabItem {
                        Label("Simulation", systemImage: "chart.line.uptrend.xyaxis")
                    }
                    .tag(2)

                StreaksView()
                    .tabItem {
                        Label("Streaks", systemImage: "flame.fill")
                    }
                    .tag(3)

                SettingsView()
                    .tabItem {
                        Label("Settings", systemImage: "gearshape.fill")
                    }
                    .tag(4)
            }
            .tint(.accentBlue)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(APIClient())
}
