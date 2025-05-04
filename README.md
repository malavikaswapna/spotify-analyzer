# 🎵 Spotify Listening Pattern Analyzer 🎵

![Spotify Analyzer Banner](https://i.imgur.com/w4BOjMO.png)

> *"Because your music taste deserves its own analytics dashboard!"* ✨

## 🎧 What is This Magical Thing?

Ever wondered what your Spotify listening habits say about you? Are you a night owl who vibes to chill lo-fi beats at 3 AM? Or perhaps a morning person who gets pumped with energetic pop hits before work?

**Spotify Listening Pattern Analyzer** is your personal music detective 🕵️‍♀️ that dives deep into your Spotify data to uncover fascinating patterns about your listening habits. It's like having your own personal music psychologist - but way more fun and without the hourly rate!

## ✨ Features That Will Make You Go "Wow!"

- 📊 **Interactive Dashboards**: Colorful visualizations that make your music taste look as good as it sounds
- 🕰️ **Temporal Analysis**: Discover when you listen to music the most (spoiler: it's probably when you should be working)
- 🎭 **Mood Mapping**: Find out if you're a secret emo kid or a sunshine pop enthusiast
- 🌐 **Artist & Genre Networks**: See how your musical tastes interconnect in beautiful web visualizations
- 💿 **Smart Recommendations**: Get song suggestions based on your actual listening patterns, not just what's trending

## 🚀 Live Demo

Check out the live demo at: [https://malasi.pythonanywhere.com/](https://malasi.pythonanywhere.com/)

## 🛠️ Tech Stack

This project was built with love and:

- **Python** - Because snakes and music go well together 🐍
- **Flask** - The lightweight web framework that keeps things simple
- **Pandas & NumPy** - For data manipulation that would make a statistician blush
- **Matplotlib & Seaborn** - Creating visualizations prettier than album covers
- **Scikit-learn** - Adding that sprinkle of machine learning magic ✨
- **D3.js** - For interactive visualizations that respond to your every hover
- **Bootstrap** - Making everything look good on any device

## 🏃‍♀️ How to Run Locally

1. Clone this repository faster than you can say "vinyl record collection":
   ```bash
   git clone https://github.com/username/spotify-analyzer.git
   cd spotify-analyzer
   ```
2. Set up a virtual environment (because sharing dependencies is caring, but not in this case):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies like you're building a music playlist:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a .env file with your Spotify API credentials:
   ```bash
   SPOTIPY_CLIENT_ID=your_client_id_here
   SPOTIPY_CLIENT_SECRET=your_client_secret_here
   SPOTIPY_REDIRECT_URI=http://localhost:5000/callback
   ```
5. Run the app and start grooving:
   ```bash
   python app.py
   ```
6. Visit http://localhost:5051 and let the magic happen!

## 🚀 Deployment

This app is currently dancing on [PythonAnywhere](https://www.pythonanywhere.com/), but it can groove on any platform that supports Python and Flask!

## 📸 Screenshots

<table>
  <tr>
    <td><img src="https://i.imgur.com/o5ArzPL.png" alt="Home Page" width="100%"></td>
    <td><img src="https://i.imgur.com/7VFGz1R.png" alt="Dashboard" width="100%"></td>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/HKPGWFE.png" alt="Recommendations" width="100%"></td>
    <td><img src="https://i.imgur.com/Ot7spRz.png" alt="Genre Networks" width="100%"></td>
  </tr>
</table>

## 🤝 Contributing

Got ideas to make this cooler? Contributions are more welcome than a surprise concert ticket! Here's how:

1. Fork it like a DJ remixing a track
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a pull request and let's collaborate!

## 📝 To-Do List

- [ ] Add more mood categories beyond the basic four
- [ ] Implement playlist generation based on analysis
- [ ] Create a time machine feature to see how your taste evolved
- [ ] Add social sharing of insights (because people need to know your top 1% Beyoncé status)
- [ ] Mobile app version for on-the-go music analytics

## 🎵 A Word From Our Sponsors (Just Kidding, We Don't Have Any)

This project is not affiliated with Spotify. I just really love music data! ❤️

## 🙏 Acknowledgments

- Spotify Web API for making data dreams come true
- My caffeine supplier (a.k.a. the local coffee shop)
- The rubber duck on my desk who listened to all my debugging woes

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ and 🎵</p>

<p align="center">
  <a href="https://github.com/malavikaswapna/spotify-analyzer/stargazers">
    <img src="https://img.shields.io/github/stars/malavikaswapna/spotify-analyzer?style=social" alt="Stars">
  </a>
  <a href="https://github.com/malavikaswapna/spotify-analyzer/network/members">
    <img src="https://img.shields.io/github/forks/malavikaswapna/spotify-analyzer?style=social" alt="Forks">
  </a>
</p>
