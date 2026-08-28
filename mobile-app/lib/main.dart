import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(const EdgeAiApp());
}

class EdgeAiApp extends StatelessWidget {
  const EdgeAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Edge AI Smart Bank Transfers',
      home: const BridgeTestPage(),
    );
  }
}

class BridgeTestPage extends StatefulWidget {
  const BridgeTestPage({super.key});

  @override
  State<BridgeTestPage> createState() =>
      _BridgeTestPageState();
}

class _BridgeTestPageState
    extends State<BridgeTestPage> {

  static const platform =
      MethodChannel('edge_ai/native');

  final TextEditingController _promptController =
      TextEditingController(
    text: 'Test prompt\n\n',
  );

  String _status = 'Ready';
  bool _busy = false;

  Future<void> _call(
    String method, [
    Map<String, dynamic>? arguments,
  ]) async {

    setState(() {
      _busy = true;
      _status = 'Working...';
    });

    try {
      final result =
          await platform.invokeMethod<String>(
        method,
        arguments,
      );

      setState(() {
        _status = result ?? 'No response';
      });

    } on PlatformException catch (e) {

      setState(() {
        _status = 'Error: ${e.message}';
      });

    } finally {

      setState(() {
        _busy = false;
      });
    }
  }

  Map<String, dynamic> get _promptArgs => {
        'prompt': _promptController.text,
      };

  @override
  void dispose() {
    _promptController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Edge AI Smart Bank Transfers',
        ),
      ),

      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),

        child: Column(
          crossAxisAlignment:
              CrossAxisAlignment.stretch,

          children: [

            ElevatedButton(
              onPressed:
                  _busy ? null : () => _call('ping'),
              child:
                  const Text('Test native bridge'),
            ),

            const SizedBox(height: 12),

            ElevatedButton(
              onPressed:
                  _busy ? null : () => _call('loadModel'),
              child:
                  const Text('Load Q5 model'),
            ),

            const SizedBox(height: 12),

            ElevatedButton(
              onPressed: _busy
                  ? null
                  : () => _call('createContext'),
              child:
                  const Text('Create context'),
            ),

            const SizedBox(height: 24),

            TextField(
              controller: _promptController,
              maxLines: 4,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                labelText: 'Inference test prompt',
              ),
            ),

            const SizedBox(height: 12),

            ElevatedButton(
              onPressed: _busy
                  ? null
                  : () => _call(
                        'tokenize',
                        _promptArgs,
                      ),
              child:
                  const Text('Tokenize prompt'),
            ),

            const SizedBox(height: 12),

            ElevatedButton(
              onPressed: _busy
                  ? null
                  : () => _call(
                        'decode',
                        _promptArgs,
                      ),
              child:
                  const Text('Decode prompt'),
            ),

            const SizedBox(height: 12),

            ElevatedButton(
              onPressed: _busy
                  ? null
                  : () => _call(
                        'sampleFirstToken',
                      ),
              child:
                  const Text('Sample first token'),
            ),

            const SizedBox(height: 12),

            ElevatedButton(
              onPressed: _busy
                  ? null
                  : () => _call(
                        'generate',
                        _promptArgs,
                      ),
              child:
                  const Text('Generate full response'),
            ),

            const SizedBox(height: 24),

            const Text(
              'Status:',
              style: TextStyle(
                fontWeight: FontWeight.bold,
              ),
            ),

            const SizedBox(height: 8),

            SelectableText(_status),
          ],
        ),
      ),
    );
  }
}


