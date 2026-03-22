import SwiftUI

struct MatchCardView: View {
    let match: Match
    let onTap: () -> Void
    
    var body: some View {
        Button(action: {
            let impactFeedback = UIImpactFeedbackGenerator(style: .light)
            impactFeedback.impactOccurred()
            onTap()
        }) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(match.competition)
                            .font(.caption)
                            .foregroundColor(.neutralGray)
                        Text(match.dateFormatted)
                            .font(.caption2)
                            .foregroundColor(.gray)
                    }
                    Spacer()
                    HStack(spacing: 4) {
                        Image(systemName: "clock")
                            .font(.caption2)
                        Text(match.timeUntilMatch)
                            .font(.caption2)
                    }
                    .foregroundColor(.accentBlue)
                }
                
                HStack(spacing: 8) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(match.homeTeam)
                            .font(.system(.body, design: .default))
                            .fontWeight(.semibold)
                            .lineLimit(2)
                        
                        Text(match.awayTeam)
                            .font(.system(.body, design: .default))
                            .fontWeight(.semibold)
                            .lineLimit(2)
                    }
                    
                    Spacer()
                    
                    if let pHome = match.pHome, let pDraw = match.pDraw, let pAway = match.pAway {
                        VStack(alignment: .trailing, spacing: 4) {
                            HStack(spacing: 4) {
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(Color.accentBlue)
                                    .frame(width: CGFloat(pHome * 50), height: 20)
                                Text(String(format: "%.0f%%", pHome * 100))
                                    .font(.caption)
                                    .fontWeight(.semibold)
                            }
                            
                            HStack(spacing: 4) {
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(Color.warningYellow)
                                    .frame(width: CGFloat(pDraw * 50), height: 20)
                                Text(String(format: "%.0f%%", pDraw * 100))
                                    .font(.caption)
                                    .fontWeight(.semibold)
                            }
                            
                            HStack(spacing: 4) {
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(Color.dangerRed)
                                    .frame(width: CGFloat(pAway * 50), height: 20)
                                Text(String(format: "%.0f%%", pAway * 100))
                                    .font(.caption)
                                    .fontWeight(.semibold)
                            }
                        }
                    }
                }
                // v12: Quick stats row
                if match.btts != nil || match.o25 != nil || match.xgHome != nil {
                    Divider()
                    HStack(spacing: 12) {
                        if let xgH = match.xgHome, let xgA = match.xgAway {
                            Label(String(format: "xG: %.1f-%.1f", xgH, xgA), systemImage: "target")
                                .font(.caption2)
                                .foregroundColor(.neutralGray)
                        }
                        if let btts = match.btts {
                            Label(String(format: "BTTS: %.0f%%", btts * 100), systemImage: "arrow.left.arrow.right")
                                .font(.caption2)
                                .foregroundColor(.neutralGray)
                        }
                        if let o25 = match.o25 {
                            Label(String(format: "O2.5: %.0f%%", o25 * 100), systemImage: "chart.bar")
                                .font(.caption2)
                                .foregroundColor(.neutralGray)
                        }
                        Spacer()
                    }
                }

                // v12: Upset risk & confidence
                HStack(spacing: 8) {
                    if let conf = match.confidence {
                        HStack(spacing: 3) {
                            Image(systemName: "target")
                                .font(.caption2)
                            Text(String(format: "%.0f%%", conf * 100))
                                .font(.caption2)
                                .fontWeight(.semibold)
                        }
                        .foregroundColor(.accentBlue)
                    }

                    if let risk = match.upsetRisk, risk > 0.3 {
                        HStack(spacing: 3) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.caption2)
                            Text(match.upsetLevel.uppercased())
                                .font(.caption2)
                                .fontWeight(.bold)
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            risk > 0.6 ? Color.dangerRed.opacity(0.15) :
                            Color.warningYellow.opacity(0.15)
                        )
                        .foregroundColor(risk > 0.6 ? .dangerRed : .warningYellow)
                        .cornerRadius(4)
                    }

                    if match.hasValue, let edge = match.valueEdge {
                        HStack(spacing: 3) {
                            Image(systemName: "arrow.up.right")
                                .font(.caption2)
                            Text(String(format: "+%.1f%%", edge * 100))
                                .font(.caption2)
                                .fontWeight(.semibold)
                        }
                        .foregroundColor(.successGreen)
                    }

                    Spacer()
                }
            }
            .padding()
            .background(Color.cardBackground)
            .cornerRadius(12)
            .shadow(color: Color.black.opacity(0.1), radius: 4, x: 0, y: 2)
        }
        .foregroundColor(.primary)
    }
}

#Preview {
    MatchCardView(
        match: Match(
            matchId: 1,
            homeTeam: "Manchester City",
            awayTeam: "Chelsea",
            competition: "Premier League",
            utcDate: "2026-03-21T15:00:00",
            pHome: 0.45,
            pDraw: 0.25,
            pAway: 0.30,
            btts: 0.62,
            o25: 0.68,
            egHome: 1.8,
            egAway: 1.2,
            upsetRisk: 0.35,
            confidence: 0.72,
            modelAgreement: 0.8,
            xgHome: 1.8,
            xgAway: 1.1,
            valueEdge: 0.042
        ),
        onTap: {}
    )
    .padding()
}
