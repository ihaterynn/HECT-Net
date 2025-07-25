import 'package:flutter/material.dart';
import '../screens/home_screen.dart';
import '../screens/scan_screen.dart';
import '../screens/profile_screen.dart';
import '../screens/meal_log_screen.dart';

class BottomNavbarWidget extends StatefulWidget {
  const BottomNavbarWidget({Key? key}) : super(key: key);

  @override
  _BottomNavbarWidgetState createState() => _BottomNavbarWidgetState();
}

class _BottomNavbarWidgetState extends State<BottomNavbarWidget> {
  int _selectedIndex = 0;
  late List<Widget> _widgetOptions;

  @override
  void initState() {
    super.initState();
    _widgetOptions = <Widget>[
      HomeScreen(onGoToScan: () => _onItemTapped(1)), // Pass callback to HomeScreen
      ScanScreen(selectedDateForLog: DateTime.now()), // MODIFIED: Pass current date
      const MealLogScreen(),
      const ProfileScreen(),
    ];
  }

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: _widgetOptions.elementAt(_selectedIndex),
      ),
      bottomNavigationBar: BottomNavigationBar(
        items: <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Image.asset('assets/images/home.png', width: 28, height: 28, color: _selectedIndex == 0 ? const Color(0xFF59CE8F) : const Color(0xFF7A7A7A)),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Image.asset('assets/images/camera.png', width: 30, height: 30, color: _selectedIndex == 1 ? const Color(0xFF59CE8F) : const Color(0xFF7A7A7A)),
            label: 'Scan',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.article_outlined, size: 28, color: _selectedIndex == 2 ? const Color(0xFF59CE8F) : const Color(0xFF7A7A7A)),
            label: 'Logs',
          ),
          BottomNavigationBarItem(
            icon: Image.asset('assets/images/profile2.png', width: 28, height: 28, color: _selectedIndex == 3 ? const Color(0xFF59CE8F) : const Color(0xFF7A7A7A)),
            label: 'Profile',
          ),
        ],
        currentIndex: _selectedIndex,
        selectedItemColor: const Color(0xFF59CE8F),
        unselectedItemColor: const Color(0xFF7A7A7A),
        onTap: _onItemTapped,
        backgroundColor: Colors.white,
        type: BottomNavigationBarType.fixed, // Ensures labels are always visible
        selectedLabelStyle: const TextStyle(fontSize: 14),
        unselectedLabelStyle: const TextStyle(fontSize: 14),
        iconSize: 28, // General icon size, can be overridden by Image asset size
        elevation: 8.0, // Adds a slight shadow
        // The React Native version had specific padding for tabBarStyle, tabBarItemStyle, tabBarIconStyle.
        // Some of these are controlled by the overall height and icon/label styles in Flutter.
        // For more precise control, you might need to wrap icons or use a custom BottomAppBar.
      ),
    );
  }
} 