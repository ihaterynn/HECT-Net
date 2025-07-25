import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io' as dart_io;
import 'package:flutter/foundation.dart' show kIsWeb, Uint8List;
import 'package:percent_indicator/percent_indicator.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:camera/camera.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:intl/intl.dart';
import 'package:http_parser/http_parser.dart';
import 'package:path/path.dart' as p;

const double DAILY_CALORIE_GOAL = 2000.0;

class WebCameraCaptureScreen extends StatefulWidget {
  final CameraDescription camera;
  const WebCameraCaptureScreen({Key? key, required this.camera}) : super(key: key);

  @override
  _WebCameraCaptureScreenState createState() => _WebCameraCaptureScreenState();
}

class _WebCameraCaptureScreenState extends State<WebCameraCaptureScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(widget.camera, ResolutionPreset.medium, enableAudio: false);
    _initializeControllerFuture = _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _captureAndPop() async {
    if (!_controller.value.isInitialized || _controller.value.isTakingPicture) return;
    try {
      await _initializeControllerFuture;
      final XFile image = await _controller.takePicture();
      if (mounted) Navigator.pop(context, image);
    } catch (e) {
      print('Error taking picture in WebCameraCaptureScreen: $e');
      if (mounted) Navigator.pop(context, null);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Capture Food Photo', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        backgroundColor: const Color(0xFF59CE8F),
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF59CE8F), Color(0xFF4CAF50), Color(0xFF81C784)],
          ),
        ),
        child: FutureBuilder<void>(
          future: _initializeControllerFuture,
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.done) {
              return Stack(
                alignment: Alignment.bottomCenter,
                children: <Widget>[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(20),
                    child: CameraPreview(_controller),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(25),
                        gradient: const LinearGradient(colors: [Color(0xFF42A5F5), Color(0xFF1E88E5)]),
                        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.3), blurRadius: 10, offset: const Offset(0, 5))],
                      ),
                      child: ElevatedButton.icon(
                        icon: const Icon(Icons.camera_alt, color: Colors.white),
                        label: const Text('Capture', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
                        onPressed: _captureAndPop,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.transparent,
                          shadowColor: Colors.transparent,
                          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                        ),
                      ),
                    ),
                  ),
                ],
              );
            } else {
              return const Center(child: CircularProgressIndicator(valueColor: AlwaysStoppedAnimation<Color>(Colors.white)));
            }
          },
        ),
      ),
    );
  }
}

class ScanScreen extends StatefulWidget {
  final DateTime selectedDateForLog;
  const ScanScreen({Key? key, required this.selectedDateForLog}) : super(key: key);

  @override
  _ScanScreenState createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> with TickerProviderStateMixin {
  bool _isLoadingInference = false;
  bool _isLoadingLogging = false;
  final ImagePicker _picker = ImagePicker();
  
  XFile? _pickedImageXFile;
  String? _previewPredictedFoodName;
  double? _previewCalories, _previewCarbs, _previewProtein, _previewFats, _previewConfidence, _previewPortionRatio;

  late AnimationController _fadeController, _slideController, _scaleController, _pulseController;
  late Animation<double> _fadeAnimation, _scaleAnimation, _pulseAnimation;
  late Animation<Offset> _slideAnimation;

  @override
  void initState() {
    super.initState();
    
    _fadeController = AnimationController(duration: const Duration(milliseconds: 800), vsync: this);
    _slideController = AnimationController(duration: const Duration(milliseconds: 600), vsync: this);
    _scaleController = AnimationController(duration: const Duration(milliseconds: 400), vsync: this);
    _pulseController = AnimationController(duration: const Duration(milliseconds: 1500), vsync: this);
    
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(CurvedAnimation(parent: _fadeController, curve: Curves.easeInOut));
    _slideAnimation = Tween<Offset>(begin: const Offset(0, 0.3), end: Offset.zero).animate(CurvedAnimation(parent: _slideController, curve: Curves.easeOutBack));
    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(CurvedAnimation(parent: _scaleController, curve: Curves.elasticOut));
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.05).animate(CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut));
    
    _fadeController.forward();
    _slideController.forward();
    _scaleController.forward();
    _pulseController.repeat(reverse: true);
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _slideController.dispose();
    _scaleController.dispose();
    _pulseController.dispose();
    super.dispose();
  }

  Future<void> _takePhoto() async {
    _resetUIState(); 
    XFile? imageXFile;
    if (kIsWeb) {
      try {
        final cameras = await availableCameras();
        if (cameras.isEmpty && mounted) {
          _showErrorDialog('No cameras available.');
          return;
        }
        if (!mounted) return;
        imageXFile = await Navigator.push(context, MaterialPageRoute(builder: (context) => WebCameraCaptureScreen(camera: cameras.first)));
      } catch (e) {
        if (mounted) _showErrorDialog('Could not initialize camera for web: $e');
      }
    } else {
      try {
        await _checkAndRequestPermission(Permission.camera, 'Camera');
        imageXFile = await _picker.pickImage(source: ImageSource.camera);
      } catch (e) {
        print('Error taking photo on mobile: $e');
      }
    }

    if (imageXFile != null && mounted) {
      setState(() {
        _pickedImageXFile = imageXFile;
      });
      _scaleController.reset();
      _scaleController.forward();
    }
  }
  
  Future<void> _pickImageFromGallery() async {
    _resetUIState();
    if (!kIsWeb) {
      try {
        await _checkAndRequestPermission(Permission.photos, 'Photo Library');
      } catch (e) {
        return; 
      }
    }
    try {
      final XFile? imageXFile = await _picker.pickImage(source: ImageSource.gallery);
      if (imageXFile != null && mounted) {
        setState(() {
          _pickedImageXFile = imageXFile;
        });
        _scaleController.reset();
        _scaleController.forward();
      }
    } catch (e) {
      if (mounted) _showErrorDialog('Could not select image from gallery: $e');
    }
  }

  Future<void> _checkAndRequestPermission(Permission permission, String permissionName) async {
    if (kIsWeb) return; 
    var status = await permission.status;
    if (!status.isGranted) {
      status = await permission.request();
      if (!status.isGranted && mounted) {
        _showErrorDialog('$permissionName permission denied.');
        throw Exception('$permissionName permission denied.'); 
      }
    }
  }
  
  void _resetUIState() {
    if (mounted) {
      setState(() {
        _pickedImageXFile = null;
        _previewPredictedFoodName = null;
        _previewCalories = null;
        _previewCarbs = null;
        _previewProtein = null;
        _previewFats = null;
        _previewConfidence = null;
        _previewPortionRatio = null;
        _isLoadingInference = false;
        _isLoadingLogging = false;
      });
    }
  }

  Future<void> _performInference() async {
    if (_pickedImageXFile == null) {
      if (mounted) _showErrorDialog('Please select an image first.');
      return;
    }
    if (Supabase.instance.client.auth.currentUser == null) {
      if (mounted) _showErrorDialog('You must be logged in to analyze meals.');
      return;
    }

    if (mounted) {
      setState(() {
        _isLoadingInference = true;
        _previewPredictedFoodName = null; 
        _previewCalories = null;
        _previewCarbs = null;
        _previewProtein = null;
        _previewFats = null;
        _previewConfidence = null;
        _previewPortionRatio = null;
      });
    }

    final String host = kIsWeb || dart_io.Platform.isWindows || dart_io.Platform.isLinux || dart_io.Platform.isMacOS ? "127.0.0.1" : "10.0.2.2";
    final url = 'http://$host:8000/predict/';
    
    try {
      var request = http.MultipartRequest('POST', Uri.parse(url));
      final Uint8List imageBytes = await _pickedImageXFile!.readAsBytes();
      
      String? mimeType = _pickedImageXFile!.mimeType; 
      String fileExtension = _pickedImageXFile!.name.split('.').last.toLowerCase();

      if (mimeType == null || mimeType.isEmpty || mimeType == "application/octet-stream") {
        if (fileExtension == 'jpg' || fileExtension == 'jpeg') mimeType = 'image/jpeg';
        else if (fileExtension == 'png') mimeType = 'image/png';
        else mimeType = 'image/jpeg';
      }
      
      request.files.add(http.MultipartFile.fromBytes('file', imageBytes, filename: _pickedImageXFile!.name, contentType: MediaType.parse(mimeType)));
      
      print('Attempting to infer from $url with image: ${_pickedImageXFile!.name} and contentType: $mimeType');
      var streamedResponse = await request.send().timeout(const Duration(seconds: 60)); 
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final responseData = json.decode(response.body);
        final prediction = responseData['prediction'];
        if (prediction != null) {
          final nutritionInfo = prediction['nutrition_info'] ?? {};
          setState(() {
            _previewPredictedFoodName = _formatFoodName(prediction['label']); // Apply formatting here
            _previewConfidence = prediction['confidence']?.toDouble();
            _previewCalories = nutritionInfo['calories']?.toDouble() ?? 0.0;
            _previewCarbs = nutritionInfo['carbs']?.toDouble() ?? 0.0;
            _previewProtein = nutritionInfo['protein']?.toDouble() ?? 0.0;
            _previewFats = nutritionInfo['fat']?.toDouble() ?? 0.0;
            _previewPortionRatio = 1.0;
          });
          _scaleController.reset();
          _scaleController.forward();
        } else {
          throw Exception('No prediction data in response');
        }
      } else {
        throw Exception('Server error: ${response.statusCode}');
      }
    } catch (e, s) { 
      print('Exception during inference with $url: $e\nStack: $s');
      if(mounted) _showErrorDialog('Failed to get data from inference server. ${e.toString()}');
    } finally {
      if (mounted) setState(() => _isLoadingInference = false);
    }
  }

  Future<void> _logMealToSupabase() async {
    if (_pickedImageXFile == null || Supabase.instance.client.auth.currentUser == null || _previewPredictedFoodName == null || _previewCalories == null) {
      if (mounted) _showErrorDialog('No valid meal data to log. Please select and analyze an image first.');
      return;
    }
  
    if (mounted) setState(() => _isLoadingLogging = true);
  
    const String bucketName = 'meal-images'; 
  
    try {
      final userId = Supabase.instance.client.auth.currentUser!.id;
      final logDate = DateFormat('yyyy-MM-dd').format(widget.selectedDateForLog);
      
      final imageBytes = await _pickedImageXFile!.readAsBytes();
      final originalFileName = _pickedImageXFile!.name;
      final fileExt = p.extension(originalFileName).isNotEmpty ? p.extension(originalFileName) : '.jpg'; 
      final fileName = '${DateTime.now().millisecondsSinceEpoch}_${p.basenameWithoutExtension(originalFileName)}$fileExt';
      
      final String imagePathForDb = '$userId/$fileName'; 
      
      await Supabase.instance.client.storage.from(bucketName).uploadBinary(
        imagePathForDb, 
        imageBytes,
        fileOptions: FileOptions(cacheControl: '3600', contentType: _pickedImageXFile!.mimeType ?? 'image/jpeg'),
      );
      print('Image uploaded successfully to path: $imagePathForDb');
  
      await Supabase.instance.client.from('meal_logs').insert({
        'user_id': userId,
        'log_date': logDate,
        'food_name': _previewPredictedFoodName,
        'calories': _previewCalories,
        'carbs': _previewCarbs,
        'protein': _previewProtein,
        'fats': _previewFats,
        'confidence': _previewConfidence, 
        'image_url': imagePathForDb,
      });
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('$_previewPredictedFoodName logged successfully! 🎉', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            backgroundColor: const Color(0xFF59CE8F),
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
        );
        _resetUIState();
        if (Navigator.canPop(context)) Navigator.pop(context, true);
      }
    } catch (e, s) {
      print("Error during meal logging or image upload: $e\nStack: $s");
      if(mounted){
        String errorMessage = 'Error saving meal or uploading image: ${e.toString()}';
        if (e is StorageException) errorMessage = 'Storage Error: ${e.message}. Intended bucket: "$bucketName"';
        _showErrorDialog(errorMessage);
      }
    } finally {
      if (mounted) setState(() => _isLoadingLogging = false);
    }
  }

  void _showErrorDialog(String message) {
    if (mounted) {
      showDialog(
        context: context,
        builder: (ctx) => AlertDialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          title: const Text('Error', style: TextStyle(color: Color(0xFF3C3F4D), fontWeight: FontWeight.bold)),
          content: SingleChildScrollView(child: Text(message, style: const TextStyle(color: Color(0xFF3C3F4D)))),
          actions: <Widget>[
            TextButton(
              child: const Text('Okay', style: TextStyle(color: Color(0xFF59CE8F), fontWeight: FontWeight.bold)),
              onPressed: () => Navigator.of(ctx).pop(),
            )
          ],
        ),
      );
    }
  }

  Widget _buildImagePreviewWidget() {
    if (_pickedImageXFile == null) return const SizedBox.shrink();
    final errorBuilder = (BuildContext c, Object e, StackTrace? s) => Center(child: Text("Error loading image: ${e.toString()}"));
    
    return kIsWeb
        ? Image.network(_pickedImageXFile!.path, fit: BoxFit.cover, errorBuilder: errorBuilder)
        : Image.file(dart_io.File(_pickedImageXFile!.path), fit: BoxFit.cover, errorBuilder: errorBuilder);
  }

  Widget _buildNutrientDisplay(String label, double? value, String unit, IconData icon, Color color) {
    return ScaleTransition(
      scale: _scaleAnimation,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.9),
          borderRadius: BorderRadius.circular(16),
          boxShadow: [BoxShadow(color: const Color(0xFF59CE8F).withOpacity(0.1), blurRadius: 8, offset: const Offset(0, 4))],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: color, size: 24),
            const SizedBox(height: 6),
            Text('${value?.toStringAsFixed(1) ?? '0.0'}$unit', style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14, color: Color(0xFF3C3F4D))),
            Text(label, style: TextStyle(fontSize: 11, color: Colors.grey[600], fontWeight: FontWeight.w500)),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    bool canUserInteract = Supabase.instance.client.auth.currentUser != null;
    bool isAnalyzingOrLogging = _isLoadingInference || _isLoadingLogging;

    return Scaffold(
      appBar: AppBar(
        title: Text('Log Meal for ${DateFormat.yMMMMd().format(widget.selectedDateForLog)}', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 18)),
        backgroundColor: const Color(0xFF59CE8F),
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.white),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: isAnalyzingOrLogging ? null : () {
            _resetUIState();
            if (Navigator.canPop(context)) Navigator.pop(context);
            else Navigator.pushReplacementNamed(context, '/meal_log');
          },
        ),
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFFCDFADB), Color(0xFFE8F5E8), Color(0xFFF0FFF0), Colors.white],
            stops: [0.0, 0.3, 0.7, 1.0],
          ),
        ),
        child: SafeArea(
          child: FadeTransition(
            opacity: _fadeAnimation,
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: <Widget>[
                  // Image preview section
                  SlideTransition(
                    position: _slideAnimation,
                    child: ScaleTransition(
                      scale: _scaleAnimation,
                      child: _pickedImageXFile != null && !isAnalyzingOrLogging
                          ? Container(
                              height: MediaQuery.of(context).size.height * 0.25,
                              margin: const EdgeInsets.only(bottom: 20),
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(20),
                                boxShadow: [BoxShadow(color: const Color(0xFF59CE8F).withOpacity(0.3), blurRadius: 15, offset: const Offset(0, 8))],
                              ),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(20),
                                child: _buildImagePreviewWidget(),
                              ),
                            )
                          : _pickedImageXFile == null && !isAnalyzingOrLogging && _previewPredictedFoodName == null
                              ? Container(
                                  height: MediaQuery.of(context).size.height * 0.25,
                                  margin: const EdgeInsets.only(bottom: 20),
                                  decoration: BoxDecoration(
                                    color: Colors.white.withOpacity(0.8),
                                    borderRadius: BorderRadius.circular(20),
                                    border: Border.all(color: const Color(0xFF59CE8F).withOpacity(0.3), width: 2),
                                    boxShadow: [BoxShadow(color: const Color(0xFF59CE8F).withOpacity(0.1), blurRadius: 10, offset: const Offset(0, 5))],
                                  ),
                                  child: Center(
                                    child: Column(
                                      mainAxisAlignment: MainAxisAlignment.center,
                                      children: [
                                        AnimatedBuilder(
                                          animation: _pulseAnimation,
                                          builder: (context, child) => Transform.scale(
                                            scale: _pulseAnimation.value,
                                            child: Icon(Icons.image_search_outlined, size: 80, color: const Color(0xFF59CE8F).withOpacity(0.7)),
                                          ),
                                        ),
                                        const SizedBox(height: 12),
                                        const Text('Select an image to analyze', style: TextStyle(color: Color(0xFF3C3F4D), fontSize: 16, fontWeight: FontWeight.w500)),
                                      ],
                                    ),
                                  ),
                                )
                              : const SizedBox.shrink(),
                    ),
                  ),

                  // Loading indicator
                  if (isAnalyzingOrLogging)
                    SlideTransition(
                      position: _slideAnimation,
                      child: Container(
                        padding: const EdgeInsets.symmetric(vertical: 30.0),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.9),
                          borderRadius: BorderRadius.circular(20),
                          boxShadow: [BoxShadow(color: const Color(0xFF59CE8F).withOpacity(0.2), blurRadius: 15, offset: const Offset(0, 8))],
                        ),
                        child: Column(
                          children: [
                            const CircularProgressIndicator(valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF59CE8F)), strokeWidth: 3),
                            const SizedBox(height: 20),
                            Text(
                              _isLoadingInference ? "🔍 Analyzing your food..." : "📝 Logging your meal...",
                              textAlign: TextAlign.center,
                              style: const TextStyle(fontSize: 16, color: Color(0xFF3C3F4D), fontWeight: FontWeight.w500),
                            ),
                          ],
                        ),
                      ),
                    ),

                  // Results card
                  if (!isAnalyzingOrLogging && _previewPredictedFoodName != null)
                    SlideTransition(
                      position: _slideAnimation,
                      child: ScaleTransition(
                        scale: _scaleAnimation,
                        child: Container(
                          margin: const EdgeInsets.symmetric(vertical: 10.0),
                          padding: const EdgeInsets.all(20.0),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.95),
                            borderRadius: BorderRadius.circular(20),
                            boxShadow: [BoxShadow(color: const Color(0xFF59CE8F).withOpacity(0.2), blurRadius: 20, offset: const Offset(0, 10))],
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              ShaderMask(
                                shaderCallback: (bounds) => const LinearGradient(colors: [Color(0xFF59CE8F), Color(0xFF4CAF50), Color(0xFF81C784)]).createShader(bounds),
                                child: Text(_previewPredictedFoodName!, style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white), textAlign: TextAlign.center),
                              ),
                              if (_previewConfidence != null)
                                Container(
                                  margin: const EdgeInsets.only(top: 8),
                                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                                  decoration: BoxDecoration(color: const Color(0xFF59CE8F).withOpacity(0.1), borderRadius: BorderRadius.circular(12)),
                                  child: Text('Confidence: ${(_previewConfidence! * 100).toStringAsFixed(1)}%', style: const TextStyle(fontSize: 14, color: Color(0xFF3C3F4D), fontWeight: FontWeight.w600), textAlign: TextAlign.center),
                                ),
                              const SizedBox(height: 25),
                              Center(
                                child: AnimatedBuilder(
                                  animation: _pulseAnimation,
                                  builder: (context, child) => Transform.scale(
                                    scale: _pulseAnimation.value,
                                    child: CircularPercentIndicator(
                                      radius: 70.0,
                                      lineWidth: 14.0,
                                      percent: ((_previewCalories ?? 0) / DAILY_CALORIE_GOAL).clamp(0.0, 1.0),
                                      center: Column(
                                        mainAxisAlignment: MainAxisAlignment.center,
                                        children: [
                                          Text("${_previewCalories?.toStringAsFixed(0) ?? '0'}", style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 20, color: Color(0xFF59CE8F))),
                                          const Text("kcal", style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14, color: Color(0xFF3C3F4D))),
                                        ],
                                      ),
                                      progressColor: const Color(0xFF59CE8F),
                                      backgroundColor: const Color(0xFFE8F5E8),
                                      circularStrokeCap: CircularStrokeCap.round,
                                    ),
                                  ),
                                ),
                              ),
                              const SizedBox(height: 25),
                              Row(
                                mainAxisAlignment: MainAxisAlignment.spaceAround,
                                children: [
                                  _buildNutrientDisplay("Carbs", _previewCarbs, "g", Icons.bakery_dining_outlined, Colors.orange),
                                  _buildNutrientDisplay("Protein", _previewProtein, "g", Icons.egg_alt_outlined, Colors.green),
                                  _buildNutrientDisplay("Fats", _previewFats, "g", Icons.oil_barrel_outlined, Colors.blueAccent),
                                ],
                              ),
                              const SizedBox(height: 30),
                              AnimatedBuilder(
                                animation: _pulseAnimation,
                                builder: (context, child) => Transform.scale(
                                  scale: _pulseAnimation.value,
                                  child: Container(
                                    decoration: BoxDecoration(
                                      borderRadius: BorderRadius.circular(25),
                                      gradient: const LinearGradient(colors: [Color(0xFFFF7043), Color(0xFFFF5722)]),
                                      boxShadow: [BoxShadow(color: const Color(0xFFFF7043).withOpacity(0.4), blurRadius: 15, offset: const Offset(0, 8))],
                                    ),
                                    child: ElevatedButton.icon(
                                      icon: const Icon(Icons.post_add_outlined, color: Colors.white),
                                      label: const Text('Log This Meal 🍽️', style: TextStyle(fontSize: 18, color: Colors.white, fontWeight: FontWeight.bold)),
                                      onPressed: _logMealToSupabase,
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Colors.transparent,
                                        shadowColor: Colors.transparent,
                                        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(25)),
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),

                  // Action buttons
                  if (!isAnalyzingOrLogging)
                    SlideTransition(
                      position: _slideAnimation,
                      child: Padding(
                        padding: const EdgeInsets.symmetric(vertical: 30.0),
                        child: Column(
                          children: [
                            Row(
                              children: [
                                Expanded(child: _buildActionButton(icon: Icons.camera_alt, label: 'Camera', onPressed: canUserInteract ? _takePhoto : null, gradientColors: [const Color(0xFF59CE8F), const Color(0xFF4CAF50)], margin: const EdgeInsets.only(right: 12))),
                                Expanded(child: _buildActionButton(icon: Icons.photo_library, label: 'Gallery', onPressed: canUserInteract ? _pickImageFromGallery : null, gradientColors: [const Color(0xFF26A69A), const Color(0xFF00ACC1)], margin: const EdgeInsets.only(left: 12))),
                              ],
                            ),
                            if (_pickedImageXFile != null) ...[
                              const SizedBox(height: 24),
                              _buildActionButton(
                                icon: Icons.analytics,
                                label: 'Analyze Food 🔍',
                                onPressed: canUserInteract ? _performInference : null,
                                gradientColors: [const Color(0xFF7E57C2), const Color(0xFF5E35B1)],
                                isWide: true,
                              ),
                            ],
                          ],
                        ),
                      ),
                    ),

                  // Login prompt
                  if (!canUserInteract)
                    SlideTransition(
                      position: _slideAnimation,
                      child: Container(
                        margin: const EdgeInsets.only(top: 20),
                        padding: const EdgeInsets.all(20),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.9),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: const Color(0xFF59CE8F).withOpacity(0.3), width: 2),
                          boxShadow: [BoxShadow(color: const Color(0xFF59CE8F).withOpacity(0.1), blurRadius: 10, offset: const Offset(0, 5))],
                        ),
                        child: Column(
                          children: [
                            Icon(Icons.login, size: 48, color: const Color(0xFF59CE8F).withOpacity(0.7)),
                            const SizedBox(height: 12),
                            const Text('Please log in to analyze and log meals', style: TextStyle(fontSize: 16, color: Color(0xFF3C3F4D), fontWeight: FontWeight.w600), textAlign: TextAlign.center),
                          ],
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  // Helper method to reduce button code duplication
  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required VoidCallback? onPressed,
    required List<Color> gradientColors,
    EdgeInsets? margin,
    bool isWide = false,
  }) {
    return AnimatedBuilder(
      animation: _scaleAnimation,
      builder: (context, child) {
        return Transform.scale(
          scale: _scaleAnimation.value,
          child: Container(
            width: isWide ? double.infinity : null,
            margin: margin,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(25),
              gradient: LinearGradient(colors: gradientColors),
              boxShadow: [BoxShadow(color: gradientColors.first.withOpacity(0.4), blurRadius: 15, offset: const Offset(0, 8))],
            ),
            child: ElevatedButton.icon(
              icon: Icon(icon, color: Colors.white, size: 24),
              label: Text(label, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
              onPressed: onPressed,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.transparent,
                shadowColor: Colors.transparent,
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(25)),
              ),
            ),
          ),
        );
      },
    );
  }
}

// Add this helper function after the _ScanScreenState class declaration
String _formatFoodName(String foodName) {
  // Convert underscore-separated class names to proper display names
  return foodName
      .split('_')
      .map((word) => word[0].toUpperCase() + word.substring(1).toLowerCase())
      .join(' ');
}